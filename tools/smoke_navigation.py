#!/usr/bin/env python3
"""Bounded integration check on an official public HM3D val example.

This is not an episode evaluation and writes no benchmark success-rate result.
The environment/agent retain the requested 1000-step, 1-meter configuration.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'habitat-lab'),str(ROOT/'GLIP')]
os.chdir(ROOT)
os.environ.setdefault('NLTK_DATA',str(ROOT/'data/nltk_data'))
import habitat
import utils.instancenav_dataset
from PSG_Nav import PSG_Nav_Agent


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3)
    smoke_args=parser.parse_args()
    config=habitat.get_config('configs/instancenav_val20_qwen.yaml')
    config.defrost()
    config.DATASET.DATA_PATH='data/datasets/instancenav/smoke/smoke.json.gz'
    config.freeze()
    args=SimpleNamespace(dataset='instancenav', split_l=-1,split_r=-1,server_port=None,
                         visualize=False,timestamp='integration_smoke',rag_mode=3,rag_max_docs=10,
                         rag_similarity_threshold=.1,rag_caption_penalty=1.,rag_window_size=5)
    # Create the simulator first to expose asset/driver errors before loading detectors.
    with habitat.Env(config=config) as env:
        env.seed(77)
        obs=env.reset()
        agent=PSG_Nav_Agent(config,args)
        agent.simulator=SimpleNamespace(_env=env)
        agent.reset()
        for i in range(smoke_args.steps):
            action=agent.act(obs)
            obs=env.step(action)
            agent.update_metrics(env.get_metrics())
            print('SMOKE_STEP',i+1,action,flush=True)
            if env.episode_over:
                break
        print('NAVIGATION_SMOKE_OK: integration only; no full episode evaluated',flush=True)


if __name__=='__main__':main()
