#!/usr/bin/env python3
"""Full-weight GPU integration check using a prerecorded, synchronized episode slice.

No robot connection; reference commands are fed back only as recorded history.
"""
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-root',required=True)
    ap.add_argument('--components-root')
    ap.add_argument('--episode-npz',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--offload-encoders',action='store_true')
    args=ap.parse_args()
    import json,time
    import numpy as np
    import torch
    from piper_wam import PiperPolicy
    from piper_wam.preprocessing import resize_rgb
    data=np.load(args.episode_npz,allow_pickle=False)
    # Images at command ticks 0,3,...,48; commands at ticks 0,...,47.
    head,hand=data['head'],data['hand']
    commands=data['executed_actions']; instruction=str(data['instruction'].item())
    if len(head)!=17 or len(hand)!=17 or commands.shape!=(48,7):
        raise ValueError('Expected 17 frames per camera and 48 physical reference commands')
    if not np.array_equal(data['frame_action_indices'],np.arange(0,49,3)):
        raise ValueError('Expected image command ticks 0,3,...,48')
    for stream in (head,hand):
        for frame in stream:resize_rgb(frame)
    if not np.isfinite(commands).all():raise ValueError('Nonfinite reference commands')
    report={'device':torch.cuda.get_device_name(),'protocol':'reset/predict/observe/predict/reset; 25v/50a, BF16, seed42',
            'physical_robot_test':False,'calls':[]}
    start=time.monotonic()
    policy=PiperPolicy(args.model_root,components_root=args.components_root,offload_encoders=args.offload_encoders)
    torch.cuda.synchronize();report['load_seconds']=time.monotonic()-start
    try:
        def timed(label,fn):
            start=time.monotonic(); result=fn();torch.cuda.synchronize()
            report['calls'].append({'name':label,'seconds':time.monotonic()-start})
            print(json.dumps(report['calls'][-1]),flush=True)
            return result
        timed('reset',lambda:policy.reset(head[0],hand[0],instruction,seed=42))
        first=timed('predict_initial',policy.predict)
        if 'reference_latents' in data:
            encoded=[policy.initial_latent.detach().float().cpu().numpy()]
            original_encoder=policy._encode_video_chunks
            def capture(*a,**kw):
                value=original_encoder(*a,**kw);encoded.append(value.detach().float().cpu().numpy());return value
            policy._encode_video_chunks=capture
        # Same raw video stream as offline preprocessing: preserve VAE caches,
        # feed 4 frames per latent. These are reference actions, not predictions.
        timed('observe_reference_48',lambda:policy.observe(head[1:],hand[1:],commands,
              first_action_index=0,frame_action_indices=np.arange(3,49,3)))
        assert policy.observed_action_count==48 and policy.server.frame_st_id==5
        if 'reference_latents' in data:
            actual=np.concatenate(encoded,axis=2)[0]
            expected=data['reference_latents']
            report['offline_latent_max_abs_difference']=float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-3)
            policy._encode_video_chunks=original_encoder
        second=timed('predict_with_history',policy.predict)
        timed('reset_again',lambda:policy.reset(head[0],hand[0],instruction,seed=42))
        repeated=timed('predict_initial_again',policy.predict)
        for actions in (first,second,repeated):
            assert actions.shape==(48,7) and actions.dtype==np.float32 and np.isfinite(actions).all()
        # Compare the original helper only after the public streaming lifecycle;
        # it deliberately clears caches and must not interrupt observe/predict.
        text,empty=policy._text_pair
        reference=timed('original_preencoded_forecast',lambda:policy.server.rollout_autoregressive(
              policy.initial_latent,text,empty,num_future_frames=4,seed=42))
        from piper_wam.preprocessing import extract_actions
        reference_actions,_=extract_actions(reference['teacher_action'],policy.stats)
        report['original_forecast_max_abs_physical']=float(np.max(np.abs(first-reference_actions)))
        np.testing.assert_allclose(first,reference_actions,atol=1e-5,rtol=1e-5)
        delta=float(np.max(np.abs(first-repeated)))
        report['reset_repeat_max_abs_physical']=delta
        np.testing.assert_allclose(first,repeated,atol=1e-5,rtol=1e-5)
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        report['passed']=True
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        np.savez(args.output.with_suffix('.npz'),initial=first,with_history=second,repeated=repeated)
        print(json.dumps(report,indent=2),flush=True)
    finally:policy.close()

if __name__=='__main__':main()
