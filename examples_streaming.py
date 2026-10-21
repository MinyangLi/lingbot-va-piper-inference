"""Integration skeleton. Run with torchrun; supply your own acquisition/control code.

This file is intentionally not an executable robot controller.
"""
import numpy as np
from piper_wam import PiperPolicy


def start_episode(policy, initial_head_rgb, initial_hand_rgb, instruction):
    policy.reset(initial_head_rgb,initial_hand_rgb,instruction,seed=42)
    return policy.predict()  # float32 [48,7], already physical absolute targets


def update_and_predict(policy, head_frames, hand_frames, issued_commands, first_action_index):
    """Use ONLY actual frames and commands from the elapsed interval.

    For N=12, collect head/hand after commands 3,6,9,12 have completed.
    For N=48, collect after commands 3,6,...,48 have completed.
    If the controller modifies targets, pass the modified issued targets.
    The caller chooses N; it must be a multiple of 12 for this VAE alignment.
    """
    n=len(issued_commands)
    policy.observe(head_frames,hand_frames,issued_commands,
                   first_action_index=first_action_index,
                   frame_action_indices=first_action_index+np.arange(3,n+1,3))
    return policy.predict()

# Instantiate PiperPolicy once, call start_episode only at episode boundaries,
# then update_and_predict after each real interval. Call policy.close() at shutdown.
# Do not fabricate repeated frames or feed the unexecuted rest of a prediction.
