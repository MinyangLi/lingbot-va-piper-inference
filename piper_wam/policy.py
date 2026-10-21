"""Piper inference with persistent observed-video and executed-command history."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .preprocessing import ACTION_CHANNELS, resize_rgb, extract_actions, pack_history_actions, validate_history
from .runtime import add_runtime_path


class PiperPolicy:
    """One GPU and sequential calls. Only reset() starts a new episode.

    Launch with torchrun --standalone --nproc_per_node=1, matching validation.
    Encoder offloading is opt-in and changes residency, not model parameters.
    """
    def __init__(self, model_root, *, components_root=None, offload_encoders=False):
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Use one GPU/process for this inference wrapper')
        if 'MASTER_ADDR' not in os.environ and not torch.distributed.is_initialized():
            raise RuntimeError('Launch your script with torchrun --standalone --nproc_per_node=1')
        if not torch.cuda.is_available():
            raise RuntimeError('The validated inference path requires a CUDA GPU')
        self.model_root = Path(model_root)
        self.components_root = Path(components_root or model_root)
        self.stats = json.loads((self.model_root/'norm_stats.json').read_text())
        contract = json.loads((self.model_root/'piper_contract.json').read_text())
        if tuple(contract['action_channels']) != ACTION_CHANNELS or contract['prediction_actions'] != 48:
            raise ValueError('Checkpoint is not compatible with the Piper 48-action contract')
        if contract['action_semantics'] != 'absolute_joint_rad_gripper_m':
            raise ValueError('Unexpected physical action representation')
        for k in ('q01', 'q99'):
            if np.asarray(self.stats[k]).shape != (7,) or not np.isfinite(self.stats[k]).all():
                raise ValueError('Expected finite seven-channel normalization statistics')
        if np.any(np.asarray(self.stats['q99']) <= np.asarray(self.stats['q01'])):
            raise ValueError('Invalid normalization quantile range')
        expected_contract = json.loads((Path(__file__).parent/'assets/piper_contract.json').read_text())
        for key in ('cameras','action_dim','height','width','source_fps','video_stride',
                    'vae_temporal_stride','actions_per_latent','future_latents','normalization'):
            if contract.get(key) != expected_contract[key]:
                raise ValueError(f'Unsupported checkpoint contract: {key}')
        if not (self.model_root/'transformer/config.json').is_file():
            raise FileNotFoundError('Missing transformer/config.json')
        for name in ('vae', 'text_encoder', 'tokenizer'):
            if not (self.components_root/name).is_dir():
                raise FileNotFoundError(f'Missing {name}; run download_models.py first')
        add_runtime_path()
        from distributed.util import init_distributed
        from modules.utils import load_vae, load_text_encoder, load_tokenizer, WanVAEStreamingWrapper
        from wan_va.wan_va_server import VA_Server
        local = int(os.environ.get('LOCAL_RANK', '0'))
        self.device = torch.device(f'cuda:{local}')
        self.dtype = torch.bfloat16
        self._owns_process_group = not torch.distributed.is_initialized()
        if self._owns_process_group:
            init_distributed(1, local, 0)
        elif torch.distributed.get_world_size() != 1:
            raise ValueError('Existing distributed process group must have world size 1')
        low, high = [0.]*30, [0.]*30
        for j, k in enumerate(ACTION_CHANNELS):
            low[k], high[k] = self.stats['q01'][j], self.stats['q99'][j]
        cfg = SimpleNamespace(
            wan22_pretrained_model_name_or_path=str(self.model_root), save_root='.',
            param_dtype=self.dtype, local_rank=local, enable_offload=False,
            preencoded_inputs=True, save_debug=False, disable_tqdm=True,
            empty_cache_per_chunk=False, env_type='piper', snr_shift=5., action_snr_shift=1.,
            patch_size=(1,2,2), height=224, width=224,
            obs_cam_keys=['observation.images.head','observation.images.hand'],
            action_dim=30, action_per_frame=12, frame_chunk_size=4, attn_window=30,
            used_action_channel_ids=list(ACTION_CHANNELS), norm_stat={'q01':low,'q99':high},
            action_norm_method='quantiles', guidance_scale=5., action_guidance_scale=1.,
            num_inference_steps=25, action_num_inference_steps=50, video_exec_step=-1)
        try:
            self.server = VA_Server(cfg)
            self.offload_encoders = bool(offload_encoders)
            self.encoder_home = torch.device('cpu') if self.offload_encoders else self.device
            self.tokenizer = load_tokenizer(str(self.components_root/'tokenizer'))
            self.text_encoder = load_text_encoder(str(self.components_root/'text_encoder'),self.dtype,self.encoder_home).eval()
            self.vae = load_vae(str(self.components_root/'vae'),self.dtype,self.encoder_home).eval()
            # Two cache objects share the same VAE weights, but never camera history.
            self.streams = [WanVAEStreamingWrapper(self.vae) for _ in range(2)]
            self._last_instruction = None
            self._text_pair = None
            self._active = False
            self._closed = False
            self.observed_action_count = 0
            self.generator = None
        except Exception:
            self.close()
            raise

    @torch.no_grad()
    def _encode_text(self, instruction):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Use the nonempty, full task instruction')
        if self._last_instruction == instruction:
            return self._text_pair
        if self.offload_encoders:
            self.text_encoder.to(self.device)
        try:
            pair=[]
            for prompt in (instruction, ''):
                tok=self.tokenizer([prompt_clean(prompt)],padding='max_length',max_length=512,
                    truncation=True,add_special_tokens=True,return_attention_mask=True,return_tensors='pt')
                mask=tok.attention_mask.to(self.device)
                value=self.text_encoder(tok.input_ids.to(self.device),mask).last_hidden_state
                value[:,int(mask.sum()):]=0
                pair.append(value.to(self.dtype))
        finally:
            if self.offload_encoders:
                self.text_encoder.to('cpu')
                torch.cuda.empty_cache()
        self._last_instruction, self._text_pair = instruction, tuple(pair)
        return self._text_pair

    @torch.no_grad()
    def _encode_video_chunks(self, head_frames, hand_frames, *, initial=False):
        """Initial single frame, then groups of four sampled frames per view."""
        # Validate/resize all inputs before touching either temporal cache.
        resized = [[resize_rgb(frame) for frame in frames]
                   for frames in (head_frames, hand_frames)]
        count = len(resized[0])
        if count != len(resized[1]) or count < 1:
            raise ValueError('Both cameras need the same positive frame count')
        if (initial and count != 1) or (not initial and count % 4):
            raise ValueError('Encode one initial frame, then complete groups of four')
        if self.offload_encoders:
            self.vae.to(self.device)
        try:
            mean = torch.tensor(self.vae.config.latents_mean, device=self.device,
                                dtype=self.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std, device=self.device,
                               dtype=self.dtype).view(1, -1, 1, 1, 1)
            views = []
            block_size = 1 if initial else 4
            for frames, stream in zip(resized, self.streams):
                blocks = []
                for offset in range(0, count, block_size):
                    video = torch.from_numpy(np.stack(frames[offset:offset+block_size]))
                    video = video.permute(3, 0, 1, 2)[None].to(self.device, self.dtype) / 255 * 2 - 1
                    mu = stream.encode_chunk(video).chunk(2, dim=1)[0]
                    blocks.append((mu - mean) / std)
                views.append(torch.cat(blocks, dim=2))
            result = torch.cat(views, dim=-1)
            expected = (1, 48, 1 if initial else count // 4, 14, 28)
            if tuple(result.shape) != expected or not torch.isfinite(result).all():
                raise ValueError(f'Invalid encoded observations: shape {tuple(result.shape)}')
            return result
        finally:
            if self.offload_encoders:
                self.vae.to('cpu')
                torch.cuda.empty_cache()

    def _require_active(self):
        if self._closed:
            raise RuntimeError('Policy is closed')
        if not self._active:
            raise RuntimeError('Call reset(head_rgb, hand_rgb, instruction) before prediction; '
                               'after an inference failure reset/replay the real history')

    @torch.no_grad()
    def reset(self, head_rgb, hand_rgb, instruction, *, seed=42):
        """Start a new episode at action index 0. This is the only history reset.

        head_rgb/hand_rgb are synchronized initial RGB frames, before action 0.
        """
        if self._closed:
            raise RuntimeError('Policy is closed')
        # Reject malformed images before altering an existing session.
        resize_rgb(head_rgb)
        resize_rgb(hand_rgb)
        text, empty = self._encode_text(instruction)
        self._active = False
        for stream in self.streams:
            stream.clear_cache()
        self.server._reset(prompt_embeds=text, negative_prompt_embeds=empty,
                           create_save_dir=False)
        self.initial_latent = self._encode_video_chunks([head_rgb], [hand_rgb], initial=True)
        self.observed_action_count = 0
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._active = True

    @torch.no_grad()
    def observe(self, head_frames, hand_frames, executed_actions, *,
                first_action_index, frame_action_indices):
        """Commit real observations/issued commands, preserving history.

        N commands [N,7], N divisible by 12, paired with N/3 frames per camera.
        Frame indices are counts of completed commands: s+3,s+6,...,s+N.
        No predicted-but-unexecuted commands or measured joint-state substitution.
        """
        self._require_active()
        commands = validate_history(head_frames, hand_frames, executed_actions,
                                    first_action_index, frame_action_indices,
                                    expected_start=self.observed_action_count)
        # Validate image values before mutation of temporal state.
        for frames in (head_frames, hand_frames):
            for frame in frames:
                resize_rgb(frame)
        action = torch.from_numpy(pack_history_actions(commands, self.stats)).to(self.device, self.dtype)
        try:
            video = self._encode_video_chunks(head_frames, hand_frames)
            start = self.server.frame_st_id
            if start == 0:
                video = torch.cat([self.initial_latent, video], dim=2)
                # The initial observation has no preceding command block.
                action = torch.cat([torch.zeros_like(action[:, :, :1]), action], dim=2)
            # Replace imagined futures with the newly observed video/commands.
            self.server.transformer.clear_pred_cache(self.server.cache_name)
            inputs = self.server._prepare_latent_input(video, action, frame_st_id=start)
            for name, action_mode in (('latent_res_lst', False), ('action_res_lst', True)):
                self.server.transformer(self.server._repeat_input_for_cfg(inputs[name]),
                                        update_cache=2, cache_name=self.server.cache_name,
                                        action_mode=action_mode)
            self.server.frame_st_id += video.shape[2]
            self.observed_action_count += len(commands)
        except Exception:
            # The cache may have been partly updated: do not continue a corrupted session.
            self._active = False
            raise

    @torch.no_grad()
    def predict_with_details(self):
        """Predict the next 48 commands; never advance observed time or reset history."""
        self._require_active()
        start = self.server.frame_st_id
        chunks = []
        try:
            self.server.transformer.clear_pred_cache(self.server.cache_name)
            # The first chunk contains the dummy initial block (3 future blocks).
            # One extra chunk supplies the 4th future block; subsequent forecasts
            # start after actual history and need just one four-block chunk.
            num_chunks = 2 if start == 0 else 1
            for offset in range(num_chunks):
                action, _ = self.server._infer(
                    frame_st_id=start + offset * 4,
                    init_latent=self.initial_latent if start == 0 and offset == 0 else None,
                    generator=self.generator, save_debug=False, return_normalized_action=True)
                chunks.append(action)
            raw = torch.cat(chunks, dim=2)
            if start == 0:
                raw = raw[:, :, 1:]
            actions, normalized = extract_actions(raw[0, :, :4], self.stats)
            return {'actions': actions, 'normalized_actions': normalized,
                    'first_action_index': self.observed_action_count}
        except Exception:
            self._active = False
            raise

    def predict(self):
        """Return float32 [48,7]: absolute joint targets (rad), gripper opening (m)."""
        return self.predict_with_details()['actions']

    def close(self):
        self._active = False
        self._closed = True
        self._text_pair = None
        self.initial_latent = None
        self.generator = None
        self.server = self.text_encoder = self.vae = self.streams = None
        if self._owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self._owns_process_group = False
