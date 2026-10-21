#!/usr/bin/env python3
"""Single-start smoke example: predict from two initial RGB images; no robot connection."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',required=True)
    p.add_argument('--components-root',help='Optional existing base model folder containing vae/text_encoder/tokenizer')
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--hand',type=Path,required=True)
    p.add_argument('--instruction',required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/actions.npz'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--offload-encoders',action='store_true',help='Move encoders to CPU between uses; requires host RAM and adds transfer time')
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    from piper_wam import PiperPolicy
    from piper_wam.preprocessing import ACTION_CHANNELS,ACTION_UNITS
    with Image.open(a.head) as im:head=np.asarray(im.convert('RGB'))
    with Image.open(a.hand) as im:hand=np.asarray(im.convert('RGB'))
    policy=PiperPolicy(a.model_root,components_root=a.components_root,offload_encoders=a.offload_encoders)
    try:
        policy.reset(head, hand, a.instruction, seed=a.seed)
        result=policy.predict_with_details()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('wb') as f:
            np.savez(f,**result,action_channels=np.array(ACTION_CHANNELS),action_units=np.array(ACTION_UNITS))
        print(f'Saved {a.output}: actions shape {result["actions"].shape}; units rad/rad/rad/rad/rad/rad/m')
    finally:
        policy.close()


if __name__=='__main__':main()
