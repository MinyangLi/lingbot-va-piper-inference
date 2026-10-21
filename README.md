# LingBot-VA Piper inference

Inference code for the **Piper four-task LingBot-VA teacher at step 30,000**.

**Weights:** [wkrfugs/lingbot-va-piper-4task-step30000](https://huggingface.co/wkrfugs/lingbot-va-piper-4task-step30000)

This teacher was fine-tuned from LingBot-VA base with video/action flow matching.
The package includes the required runtime, two-camera preprocessing, persistent
history handling, and action conversion. It is not an OpenPI checkpoint or a
Flash-WAM/DriftWAM distilled model. It does not send commands to a robot.

## Install

Use Linux, Python 3.10+, a CUDA-capable PyTorch installation and one GPU.
The reference environment uses PyTorch 2.9.0, torchvision 0.24.0,
Diffusers 0.36.0 and Transformers 4.55.2.

```bash
git clone https://github.com/MinyangLi/lingbot-va-piper-inference.git
cd lingbot-va-piper-inference
pip install -e .
```

The default inference backend is PyTorch scaled dot-product attention.
**FlashAttention is not required** for this backend. It is optional only if
modifying the runtime to select `flashattn`. No separate LingBot-VA source checkout
or manual patch is needed. Use a dedicated environment/process because upstream
imports include generic module names such as `modules`, `utils` and `distributed`.

## Download weights

```bash
python download_models.py --model-root models/piper_teacher --max-workers 2
```

The script pins:

- Piper transformer: `bc103950dd60ba56793a1f59b2d680e5bd6a7af9` in the weights repo above.
- VAE, text encoder and tokenizer: `robbyant/lingbot-va-base` at
  `68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c`.

It excludes the base transformer's weights. The Piper FP32 weight file is
20.36 GB (18.96 GiB); frozen components add 14.20 GB. Compute uses BF16.
An existing copy of the pinned base components can be reused with
`components_root=...` / `--components-root`; the download script also accepts
`--skip-base-components` when those components already exist elsewhere.

## Public API: reset → predict → observe → predict

Launch the caller with **`torchrun --standalone --nproc_per_node=1`**.
Keep one policy instance for the session. Calls must be sequential.

```python
import numpy as np
from piper_wam import PiperPolicy

policy = PiperPolicy('models/piper_teacher')
try:
    # Start of a real episode: synchronized RGB images BEFORE command 0.
    policy.reset(initial_head_rgb, initial_hand_rgb,
                 instruction='Pick the pen and put it in the cup.', seed=42)
    actions = policy.predict()     # float32 [48,7], ALREADY physical units

    # The controller decides how many commands to execute. Example: 12.
    # Actually execute that interval and collect synchronized images at
    # completed-command counts 3,6,9,12. Supply actual issued commands,
    # including any controller-side modifications, NOT unused predictions.
    policy.observe(
        head_frames=head_after_3_6_9_12,
        hand_frames=hand_after_3_6_9_12,
        executed_actions=actually_issued_commands,  # [12,7], physical units
        first_action_index=0,
        frame_action_indices=np.array([3,6,9,12]),
    )
    actions = policy.predict()     # predicts commands 12..59 using real history

    # For the next 12-command interval: first_action_index=12,
    # frame_action_indices=[15,18,21,24]. Repeat observe/predict as needed.
    # Call reset(...) only when beginning another episode.
finally:
    policy.close()
```

The names representing images/control in this example are caller-provided values,
not functions implemented here. `examples_streaming.py` provides an integration
skeleton. The collaborator owns acquisition, robot transport, feedback and control.

**History is preserved.** Each camera has a separate causal VAE cache; the two
cache objects share VAE weights. Real observations and issued commands populate
persistent transformer key/value caches. On new observations, imagined-future
entries are discarded and replaced with actual history. Prediction does not
advance observed time. Only `reset()` clears the episode. The upstream bounded
transformer cache/attention window is retained; history is not unlimited.

If a model/device failure partly updates the cache, that session becomes invalid;
reset and replay actual history instead of continuing a partially updated cache.
There is no automatic episode reset, synthetic history padding or implicit skipping.

## Exact input/output contract

### Cameras and text

- Two synchronized cameras: **head, then hand** (`observation.images.head`,
  `observation.images.hand`). Inputs are `uint8` RGB `[H,W,3]`, **not OpenCV BGR**.
- Each view is independently resized with PIL bilinear interpolation, preserving
  aspect ratio, and centered black-padded to **224×224**. A 640×480 image becomes
  224×168 content. This is the preprocessing used for this Piper checkpoint.
- Each view is VAE-encoded separately, using the posterior mean and latent mean/std.
  Latents are concatenated along width, head then hand: `[1,48,F,14,28]`.
  Do not concatenate raw RGB views or insert a third-camera placeholder.
- Use the full original task instruction. Text padding embeddings are zeroed.
- This is not an OpenPI policy wrapper: matching camera names alone does not make
  its normalization, temporal interface or output layout interchangeable.

### Actions

`predict()` returns **`[48,7]` already unnormalized**:

```text
joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper
  rad      rad      rad      rad      rad      rad      m
```

These are **absolute joint and gripper targets**, matching the source dataset's `action`.
They are not Cartesian poses, velocities, deltas or measured joint states.
Only zero-based model channels **`[14,15,16,17,18,19,28]`** are used.
Other model channels are ignored. The wrapper performs channel selection and
unnormalization exactly once:

```python
physical = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

`predict_with_details()` additionally returns `normalized_actions` and
`first_action_index`. Do not unnormalize `actions` again. Use the model's
`norm_stats.json`, fitted on the 113 training episodes without clipping labels.
These quantiles are not hardware limits; do not substitute OpenPI statistics.

`observe()` also takes physical units: pass the actual absolute target commands
issued during the interval, not the whole previously predicted 48-action array
unless all 48 were issued. Do not substitute `observation.state` or the unused
prediction tail. This follows the teacher's command-history semantics.

### Time alignment — required for observed history

Training uses **30 Hz command targets** and **10 FPS video** (every third source
frame). The causal VAE consumes one initial image, then groups of four new images.
Each later video latent aligns with the **preceding 12 commands**.

| Observation | Image command-tick indices | Associated commands (zero-based) |
|---|---|---|
| Initial latent | `0` | No command block; normalized dummy zeros |
| First later latent | `3,6,9,12` | `0..11` |
| Second later latent | `15,18,21,24` | `12..23` |
| Third later latent | `27,30,33,36` | `24..35` |

For each `observe()` call starting at command `s`:

- Supply `N` **consecutive actual commands**, with `N` a positive multiple of 12.
- Supply `N/3` synchronized images per camera, sampled after completed-command
  counts `s+3,s+6,...,s+N`. Do not include the initial or previous boundary image again.
- `first_action_index=s` must match the policy's current observed-command count.
- `frame_action_indices` describes those completed-command counts, **not image-array
  offsets**. Checks reject skipped/repeated intervals and incorrect declared alignment.
  The caller remains responsible for real acquisition timestamps and synchronization.

The caller chooses how many targets to execute; a 48-target forecast is not an
instruction to execute all 48 open-loop. This adapter accepts feedback at complete
12-command boundaries. Buffer shorter real intervals until a full block is available;
do not repeat frames or commands to fake one. A higher-frequency feedback protocol
would need a separately designed/validated adapter.

At the original 30 Hz, 48 commands span 1.6 seconds. If hardware cannot maintain
that timing, this package does not silently resample targets or compensate for
inference latency. The controller must explicitly handle timing and stale commands.

### Inference settings

Teacher sampling: **25 video / 50 action denoising steps**, guidance **5/1**, SNR
shifts **5/1**, frame chunk 4, attention window 30, BF16 compute. These are model
sampling settings, not execution frequencies. A fresh first chunk contains an
initial dummy block, so the wrapper generates another chunk and returns the first
48 valid future actions. Later forecasts follow the actual accumulated history.
The Piper/OpenPI reference horizon is 50; this model's forecast is 48.

Optional `offload_encoders=True` / `--offload-encoders` moves encoder weights to
GPU for encoding and back to CPU afterward. It costs transfers and host RAM;
causal feature caches remain alive. It is not a guarantee of fitting a 24GB card.

## One-shot smoke command

```bash
torchrun --standalone --nproc_per_node=1 infer_images.py \
  --model-root models/piper_teacher \
  --head /path/to/head.png --hand /path/to/hand.png \
  --instruction 'Pick the pen and put it in the cup.' \
  --output outputs/actions.npz --seed 42
```

This command intentionally starts a new episode and makes **one** prediction.
Do not repeatedly launch it for closed-loop operation; use the persistent API above.
The NPZ contains physical `actions`, diagnostic `normalized_actions`,
`first_action_index`, `action_channels` and `action_units`.

## Verification

CPU tests include the real transformer/server/cache code with a small randomly
initialized CPU model, plus interface tests. These check camera separation and
history retention, action/time mapping, predicted-vs-observed caches, reset
reproducibility, text masking, normalization and invalid-session handling.

```bash
python -m unittest discover -s tests -v
```

The full-weight GPU integration script uses a **prerecorded episode**, never a robot:

```bash
torchrun --standalone --nproc_per_node=1 verify_inference.py \
  --model-root models/piper_teacher \
  --episode-npz /path/to/recorded_slice.npz \
  --output outputs/gpu_verification.json
```

Fixture keys: `head`/`hand` RGB arrays `[17,H,W,3]` at command ticks `0,3,...,48`,
`executed_actions` physical `[48,7]` from ticks `0..47`, `instruction` scalar string,
`frame_action_indices=[0,3,...,48]`. Optional `reference_latents=[48,5,14,28]`
compares against the original offline encoder. The fixture is not included because
it contains dataset images. Reports contain timing, shape/finite checks, original
sampler equivalence and episode-reset reproducibility. See `VERIFICATION.md` for
the actual completed checks and remaining limits; passing is not a robot success rate.

## Training context

113 train / 12 held-out episodes, four tasks, 30,000 updates, effective batch 32,
video/action FM weights 1/1. Training action L1 was a diagnostic, not a loss term.

| Final metric | Value |
|---|---:|
| Video FM, last 100-update mean | 0.00614567 |
| Action FM, last 100-update mean | 0.00033954 |
| Training denoising action L1, last 100-update mean | 0.00373420 |
| Held-out rollout normalized L1 (48 actions) | 0.08967018 |
| Held-out joint MAE | 4.27345 degrees |
| Held-out gripper MAE | 3.83985 mm |

Held-out validation used 36 conditions: 12 episodes × three phases. Its cached
latents included causal visual history, but each sampled forecast started a new
transformer context. It did not measure the streaming closed-loop adapter's robot
success rate. Training denoising L1 and sampled-rollout L1 are different metrics.
The model's physical performance and deployment latency must be measured on hardware.

## Attribution

Runtime derived from [Robbyant/LingBot-VA](https://github.com/Robbyant/lingbot-va),
commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`, with the local model/server fixes
used for teacher validation. Distribution changes remove unrelated example configs
and make the unused FlashAttention import optional. Sampling equations and weights
are unchanged. `vendor_manifest.json` records included-file hashes and changes.

Apache-2.0 notices are preserved in `LICENSE` and `NOTICE`. No raw dataset,
credentials, model weights, optimizer state or robot-control client is in this repo.
