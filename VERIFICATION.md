# Verification record — 2026-10-21

## Completed

- Ten CPU tests passed, including actual server/transformer/cache execution on a
  small random model and actual causal VAE camera-cache interleaving.
- Full step30000 teacher weights tested on **one NVIDIA A800 80GB PCIe**.
- A prerecorded validation episode provided two-camera images and real reference
  commands. No robot was connected and no training ran.
- Public lifecycle exercised: reset → predict → observe 48 recorded commands and
  16 new images per camera → predict → reset → predict again.
- Returned actions were finite float32 `[48,7]` physical-unit targets.
- Initial prediction matched the original preencoded rollout exactly:
  maximum absolute physical-action difference **0.0**.
- The initial + four subsequent streamed video latents matched the original saved
  offline preprocessing exactly: maximum absolute difference
  **0.0**.
- Reset with the same inputs/seed reproduced the initial prediction exactly:
  maximum absolute physical-action difference **0.0**.

## Observed performance on this check

| Operation | Time |
|---|---:|
| Initial forecast (two internal chunks) | 14.68 s |
| Encode/commit 48-command reference interval | 0.95 s |
| Forecast with observed history (one chunk) | 7.37 s |
| Peak GPU memory allocated | 26.83 GiB |
| Peak GPU memory reserved | 27.30 GiB |

Model loading is separate and storage-dependent. This check uses full teacher
25v/50a sampling, guidance5/1, SNR5/1, BF16, both encoders resident on GPU.
It **does not provide real-time 30 Hz policy refresh**. Thirty Hz is the target
sequence's sampling rate. The controller must explicitly handle inference latency;
this repository implements no asynchronous robot controller.

## Limits

- This is an integration/reproducibility check on one recorded slice, not a robot
  success-rate evaluation or proof of long-episode robustness.
- Encoder offload is available but not part of the full-weight GPU measurement.
- 24GB GPU fit and other GPU/driver combinations have not been verified.
- Physical units, acquisition timing, joint ordering and the actual executed
  targets must be respected by the integrating robot controller.
- Offline validation errors from training used a fresh transformer context and
  causal visual latents. They are not streaming-deployment success rates.

Machine-readable measurements and exact tested code hashes:
[`verification/gpu_report.json`](verification/gpu_report.json).
