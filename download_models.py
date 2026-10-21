#!/usr/bin/env python3
"""Download pinned teacher weights and frozen encoders; no GPU needed."""
import argparse
from huggingface_hub import snapshot_download

TEACHER_REPO='wkrfugs/lingbot-va-piper-4task-step30000'
TEACHER_REVISION='bc103950dd60ba56793a1f59b2d680e5bd6a7af9'
BASE_REPO='robbyant/lingbot-va-base'
BASE_REVISION='68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',default='models/piper_teacher')
    p.add_argument('--max-workers',type=int,default=2)
    p.add_argument('--skip-base-components',action='store_true',help='Only download Piper transformer/metadata; reuse existing pinned encoders with --components-root at inference')
    a=p.parse_args()
    snapshot_download(TEACHER_REPO,revision=TEACHER_REVISION,local_dir=a.model_root,
        allow_patterns=['transformer/*','norm_stats.json','piper_contract.json'],max_workers=a.max_workers)
    if not a.skip_base_components:
        snapshot_download(BASE_REPO,revision=BASE_REVISION,local_dir=a.model_root,
            allow_patterns=['vae/*','text_encoder/*','tokenizer/*'],max_workers=a.max_workers)
    print(f'Requested model files downloaded to {a.model_root}')


if __name__=='__main__':main()
