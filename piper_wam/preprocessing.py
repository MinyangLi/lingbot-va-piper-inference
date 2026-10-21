"""Exact image and action conventions used for the Piper teacher."""
import numpy as np
from PIL import Image

ACTION_CHANNELS = (14, 15, 16, 17, 18, 19, 28)
ACTION_UNITS = ('rad', 'rad', 'rad', 'rad', 'rad', 'rad', 'm')


def resize_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 RGB [H,W,3]; convert OpenCV BGR explicitly')
    image = Image.fromarray(rgb)
    scale = min(224 / image.width, 224 / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    if min(size) < 1:
        raise ValueError('Image aspect ratio is too extreme for the trained resize rule')
    resized = image.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (224, 224), (0, 0, 0))
    canvas.paste(resized, ((224-size[0])//2, (224-size[1])//2))
    return np.asarray(canvas)


def normalize(actions, stats):
    x = np.asarray(actions, dtype=np.float32)
    low, high = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return 2 * (x-low) / (high-low+1e-6) - 1


def denormalize(actions, stats):
    x = np.asarray(actions, dtype=np.float32)
    low, high = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return (x+1)/2 * (high-low+1e-6) + low


def extract_actions(action_tensor, stats):
    """Input normalized [30,4,12,1]; return physical and normalized [48,7]."""
    if tuple(action_tensor.shape) != (30, 4, 12, 1):
        raise ValueError('Expected normalized rollout actions [30,4,12,1]')
    normalized = action_tensor[list(ACTION_CHANNELS), :, :, 0].permute(1, 2, 0).reshape(48, 7).float().cpu().numpy()
    if not np.isfinite(normalized).all():
        raise ValueError('Model returned nonfinite actions')
    return denormalize(normalized, stats), normalized


def validate_history(head_frames, hand_frames, commands, first_action_index,
                     frame_action_indices, *, expected_start):
    """Validate the exact training-time 30 Hz action / 10 FPS image alignment."""
    x = np.asarray(commands, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 7 or len(x) < 12 or len(x) % 12:
        raise ValueError('executed_actions must be [N,7], with positive N divisible by 12')
    if not np.isfinite(x).all():
        raise ValueError('executed_actions contains nonfinite values')
    if not isinstance(first_action_index, (int, np.integer)) or first_action_index != expected_start:
        raise ValueError(f'History must start at action index {expected_start}; no gaps or duplicates')
    nframes = len(x) // 3
    if len(head_frames) != nframes or len(hand_frames) != nframes:
        raise ValueError(f'Need {nframes} synchronized frames per camera for {len(x)} commands')
    indices = np.asarray(frame_action_indices)
    expected = np.arange(expected_start+3, expected_start+len(x)+1, 3)
    if indices.dtype.kind not in 'iu' or not np.array_equal(indices, expected):
        raise ValueError('frame_action_indices must be completed-command counts s+3,s+6,...,s+N')
    return x


def pack_history_actions(commands, stats):
    """Physical [12*k,7] -> normalized [1,30,k,12,1]; unused channels stay zero."""
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 7 or len(commands) < 12 or len(commands) % 12:
        raise ValueError('Expected positive complete 12-command history blocks')
    if not np.isfinite(commands).all():
        raise ValueError('History commands must be finite')
    value = normalize(commands, stats).reshape(-1, 12, 7).transpose(2, 0, 1)
    packed = np.zeros((30, len(commands)//12, 12, 1), dtype=np.float32)
    packed[list(ACTION_CHANNELS), :, :, 0] = value
    return packed[None]
