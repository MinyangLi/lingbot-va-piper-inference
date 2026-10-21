"""CPU contract tests; no weights, downloads or robot connection."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from piper_wam.preprocessing import resize_rgb, normalize, denormalize, extract_actions, ACTION_CHANNELS
from piper_wam.policy import PiperPolicy

ROOT = Path(__file__).resolve().parents[1]

class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.stats = json.loads((ROOT/'piper_wam/assets/norm_stats.json').read_text())

    def test_resize_rgb_geometry_and_color(self):
        source=np.zeros((480,640,3),np.uint8)
        source[:,:,0]=255
        out=resize_rgb(source)
        self.assertEqual(out.shape,(224,224,3))
        self.assertTrue((out[:28]==0).all())
        self.assertTrue((out[196:]==0).all())
        np.testing.assert_array_equal(out[28:196,:,0],255)
        np.testing.assert_array_equal(out[:,:,1:],0)
        with self.assertRaises(ValueError): resize_rgb(source.astype(np.float32))

    def test_unclipped_normalization_roundtrip(self):
        values=np.array([self.stats['q01'],self.stats['q99']],np.float32)
        values[0]-=1
        norm=normalize(values,self.stats)
        self.assertTrue((norm[0]<-1).all())
        np.testing.assert_allclose(denormalize(norm,self.stats),values,atol=2e-6)

    def test_action_channels_and_temporal_order(self):
        source=torch.full((30,4,12,1),999.)
        expected=np.arange(48*7,dtype=np.float32).reshape(48,7)/100
        source[list(ACTION_CHANNELS),:,:,0]=torch.from_numpy(expected).reshape(4,12,7).permute(2,0,1)
        physical,norm=extract_actions(source,self.stats)
        np.testing.assert_array_equal(norm,expected)
        np.testing.assert_allclose(physical,denormalize(expected,self.stats))
        with self.assertRaises(ValueError): extract_actions(source[:7],self.stats)
        source[14,0,0,0]=float('nan')
        with self.assertRaises(ValueError): extract_actions(source,self.stats)

    def test_full_prompt_padding_and_text_reuse(self):
        policy=PiperPolicy.__new__(PiperPolicy)
        policy.device=torch.device('cpu'); policy.dtype=torch.float32
        policy.offload_encoders=False; policy._last_instruction=None; policy._text_pair=None
        seen=[]
        def tokenizer(prompt,**kwargs):
            seen.append(prompt[0])
            self.assertEqual(kwargs['max_length'],512)
            return SimpleNamespace(input_ids=torch.ones(1,512,dtype=torch.long),
                attention_mask=torch.cat([torch.ones(1,3,dtype=torch.long),torch.zeros(1,509,dtype=torch.long)],1))
        policy.tokenizer=tokenizer
        policy.text_encoder=lambda ids,mask: SimpleNamespace(last_hidden_state=torch.ones(1,512,8))
        prompt='Pick the carrot and put it in the bowl.'
        pair=policy._encode_text(prompt)
        self.assertEqual(seen,[prompt,''])
        self.assertTrue((pair[0][:,:3]==1).all()); self.assertTrue((pair[0][:,3:]==0).all())
        self.assertIs(policy._encode_text(prompt),pair)
        self.assertEqual(len(seen),2)
        policy._encode_text('Move the bowl.')
        self.assertEqual(len(seen),4)

    def test_actual_rollout_reset_truncation_and_repeated_calls(self):
        # Execute the included method against a fake denoiser. This checks its
        # real reset/history/truncation code, without importing CUDA model modules.
        import math
        tree=ast.parse((ROOT/'piper_wam/_vendor/lingbot_va/wan_va/wan_va_server.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VA_Server')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rollout_autoregressive')
        namespace={'torch':torch,'math':math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'rollout_under_test','exec'),namespace)
        class FakeServer:
            rollout_autoregressive=namespace['rollout_autoregressive']
            device='cpu'; cache_name='pos'; action_mask=torch.ones(30)
            job_config=SimpleNamespace(frame_chunk_size=4)
            def __init__(self): self.resets=0; self.steps=[]; self.transformer=SimpleNamespace(clear_cache=Mock())
            def _reset(self,**kwargs): self.resets+=1; self.steps=[]
            def _infer(self,frame_st_id,init_latent,**kwargs):
                self.steps.append((frame_st_id,init_latent is not None))
                t=torch.arange(frame_st_id,frame_st_id+4,dtype=torch.float32)
                return t.view(1,1,4,1,1).expand(1,30,4,12,1),t.view(1,1,4,1,1).expand(1,48,4,14,28)
        server=FakeServer()
        for _ in range(2):
            result=server.rollout_autoregressive(torch.zeros(1,48,1,14,28),None,None,4,42)
            self.assertEqual(server.steps,[(0,True),(4,False)])
            self.assertEqual(tuple(result['teacher_action'].shape),(30,4,12,1))
            torch.testing.assert_close(result['teacher_action'][0,:,0,0],torch.tensor([1.,2.,3.,4.]))
        self.assertEqual(server.resets,2)
        self.assertEqual(server.transformer.clear_cache.call_count,2)

if __name__=='__main__': unittest.main()
