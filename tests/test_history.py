"""Streaming contract tests using the real server/cache code and a tiny CPU model."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from piper_wam.policy import PiperPolicy
from piper_wam.preprocessing import ACTION_CHANNELS, pack_history_actions, normalize, validate_history
from piper_wam.runtime import add_runtime_path

ROOT = Path(__file__).resolve().parents[1]

class FakeStream:
    def __init__(self): self.clears=0; self.clear_cache()
    def clear_cache(self): self.clears+=1; self.calls=[]; self.running=0.
    def encode_chunk(self, video):
        self.calls.append(video.clone())
        if len(self.calls)==1:
            assert video.shape[2]==1
        else:
            assert video.shape[2]==4
        self.running += float(video.mean())
        mu=torch.full((1,48,1,14,14),self.running)
        return torch.cat([mu,torch.zeros_like(mu)],dim=1)


def make_policy():
    add_runtime_path()
    from modules.model import WanTransformer3DModel
    from wan_va.wan_va_server import VA_Server
    from utils.scheduler import FlowMatchScheduler
    torch.manual_seed(7)
    torch.set_num_threads(2)
    p=PiperPolicy.__new__(PiperPolicy)
    p.device=torch.device('cpu'); p.dtype=torch.float32; p.offload_encoders=False
    p.stats=json.loads((ROOT/'piper_wam/assets/norm_stats.json').read_text())
    p._closed=False; p._active=False; p._owns_process_group=False
    p._last_instruction=None; p._text_pair=None
    p.vae=SimpleNamespace(config=SimpleNamespace(latents_mean=[0.]*48,latents_std=[1.]*48))
    p.streams=[FakeStream(),FakeStream()]
    p._encode_text=Mock(return_value=(torch.ones(1,3,8),torch.zeros(1,3,8)))
    s=VA_Server.__new__(VA_Server)
    s.transformer=WanTransformer3DModel(num_attention_heads=1,attention_head_dim=24,
         text_dim=8,freq_dim=16,ffn_dim=32,num_layers=1,rope_max_seq_len=1024).eval()
    s.cache_name='pos'; s.device=p.device; s.dtype=p.dtype
    s.streaming_vae=None; s.streaming_vae_half=None; s.env_type='piper'
    s.save_root='.'; s.empty_cache_per_chunk=False; s.disable_tqdm=True; s.save_debug=False
    s.scheduler=FlowMatchScheduler(shift=5.,sigma_min=0.,extra_one_step=True)
    s.action_scheduler=FlowMatchScheduler(shift=1.,sigma_min=0.,extra_one_step=True)
    low,high=[0.]*30,[0.]*30
    for j,k in enumerate(ACTION_CHANNELS):low[k]=p.stats['q01'][j];high[k]=p.stats['q99'][j]
    # Only the tiny CPU test model and denoising counts differ from production.
    s.job_config=SimpleNamespace(guidance_scale=5.,action_guidance_scale=1.,height=224,width=224,
        obs_cam_keys=['head','hand'],patch_size=(1,2,2),frame_chunk_size=4,action_per_frame=12,
        attn_window=30,action_dim=30,used_action_channel_ids=list(ACTION_CHANNELS),
        norm_stat={'q01':low,'q99':high},action_norm_method='quantiles',num_inference_steps=2,
        action_num_inference_steps=2,video_exec_step=-1)
    p.server=s
    return p

class HistoryTests(unittest.TestCase):
    def test_history_action_mapping_and_alignment(self):
        stats=json.loads((ROOT/'piper_wam/assets/norm_stats.json').read_text())
        a=np.arange(24*7,dtype=np.float32).reshape(24,7)/100
        packed=pack_history_actions(a,stats)
        self.assertEqual(packed.shape,(1,30,2,12,1))
        recovered=packed[0,list(ACTION_CHANNELS),:,:,0].transpose(1,2,0).reshape(24,7)
        np.testing.assert_allclose(recovered,normalize(a,stats))
        self.assertTrue((packed[:,[k for k in range(30) if k not in ACTION_CHANNELS]]==0).all())
        validate_history([None]*8,[None]*8,a,12,np.arange(15,37,3),expected_start=12)
        for kwargs in [dict(first_action_index=0,frame_action_indices=np.arange(15,37,3)),
                       dict(first_action_index=12,frame_action_indices=np.arange(12,36,3))]:
            with self.assertRaises(ValueError):
                validate_history([None]*8,[None]*8,a,expected_start=12,**kwargs)

    def test_streams_preserve_history_separately(self):
        p=make_policy(); black=np.zeros((224,224,3),np.uint8); white=black+255
        p.reset(black,white,'task')
        first=p.initial_latent.clone()
        second=p._encode_video_chunks([black]*4,[white]*4)
        torch.testing.assert_close(second,first*2)
        self.assertTrue((first[...,:14]==-1).all()); self.assertTrue((first[...,14:]==1).all())
        self.assertEqual([len(s.calls) for s in p.streams],[2,2])
        p.reset(white,black,'task')
        torch.testing.assert_close(p.initial_latent,-first)
        self.assertEqual([len(s.calls) for s in p.streams],[1,1])
        p.close()

    def test_real_cpu_server_predict_observe_predict_reset(self):
        p=make_policy(); image=np.full((224,224,3),128,np.uint8)
        with self.assertRaises(RuntimeError):p.predict()
        p.reset(image,image,'task',seed=42)
        first=p.predict_with_details()
        self.assertEqual(first['actions'].shape,(48,7)); self.assertTrue(np.isfinite(first['actions']).all())
        self.assertEqual(p.server.frame_st_id,0)
        commands=np.tile((np.asarray(p.stats['q01'])+np.asarray(p.stats['q99']))/2,(12,1)).astype(np.float32)
        captured=[]
        handle=p.server.transformer.register_forward_pre_hook(
            lambda module,args,kwargs: captured.append((kwargs.get('update_cache'),kwargs.get('action_mode'),
                                                       args[0]['noisy_latents'].detach().clone())),with_kwargs=True)
        p.observe([image]*4,[image]*4,commands,first_action_index=0,frame_action_indices=[3,6,9,12])
        handle.remove()
        self.assertEqual(p.server.frame_st_id,2); self.assertEqual(p.observed_action_count,12)
        action_update=next(x[2] for x in captured if x[:2]==(2,True))
        self.assertTrue((action_update[:,:,0]==0).all())
        self.assertEqual(action_update.shape,(2,30,2,12,1)) # CFG batch2, initial+new block
        cache=p.server.transformer.blocks[0].attn1.attn_caches['pos']
        observed=(cache['mask'] & ~cache['is_pred']).clone()
        self.assertTrue(observed.any()); self.assertFalse((cache['mask'] & cache['is_pred']).any())
        old_keys=cache['k'][:,observed].clone()
        second=p.predict_with_details()
        self.assertEqual(second['first_action_index'],12)
        self.assertEqual(second['actions'].shape,(48,7)); self.assertTrue(np.isfinite(second['actions']).all())
        torch.testing.assert_close(cache['k'][:,observed],old_keys)
        self.assertEqual(p.server.frame_st_id,2)
        p.observe([image]*4,[image]*4,commands,first_action_index=12,frame_action_indices=[15,18,21,24])
        self.assertEqual(p.server.frame_st_id,3)
        with self.assertRaises(ValueError):
            p.observe([image]*4,[image]*4,commands,first_action_index=12,frame_action_indices=[15,18,21,24])
        self.assertTrue(p._active)
        p.reset(image,image,'task',seed=42)
        np.testing.assert_allclose(p.predict(),first['actions'],atol=1e-6,rtol=1e-6)
        p.close()
        with self.assertRaises(RuntimeError):p.predict()

    def test_real_vae_interleaving_matches_offline_encoding(self):
        from diffusers import AutoencoderKLWan
        add_runtime_path()
        from modules.utils import WanVAEStreamingWrapper
        torch.set_num_threads(2); torch.manual_seed(3)
        vae=AutoencoderKLWan(base_dim=4,decoder_base_dim=4,z_dim=48,dim_mult=[1,2,2,2],num_res_blocks=1,
            latents_mean=[0.]*48,latents_std=[1.]*48,in_channels=12,out_channels=12,
            patch_size=2,scale_factor_spatial=16).eval()
        images=torch.randn(2,3,9,32,32)
        intervals=((0,1),(1,5),(5,9))
        with torch.no_grad():
            reference=[]
            for camera in range(2):
                stream=WanVAEStreamingWrapper(vae)
                reference.append(torch.cat([stream.encode_chunk(images[camera:camera+1,:,a:b])
                                            for a,b in intervals],dim=2))
            streams=[WanVAEStreamingWrapper(vae),WanVAEStreamingWrapper(vae)]; outputs=[[],[]]
            for a,b in intervals:
                for camera in range(2):
                    outputs[camera].append(streams[camera].encode_chunk(images[camera:camera+1,:,a:b]))
            for camera in range(2):
                torch.testing.assert_close(torch.cat(outputs[camera],2),reference[camera],atol=0,rtol=0)

    def test_partial_failure_requires_reset(self):
        p=make_policy(); image=np.zeros((224,224,3),np.uint8)
        p.reset(image,image,'task')
        p.server._infer=Mock(side_effect=RuntimeError('simulated device failure'))
        with self.assertRaises(RuntimeError):p.predict()
        self.assertFalse(p._active)
        with self.assertRaises(RuntimeError):p.predict()
        p.close()

if __name__=='__main__':unittest.main()
