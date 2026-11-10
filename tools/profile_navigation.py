#!/usr/bin/env python3
"""Measure real navigation without changing actions, detectors, or evaluation limits."""
import argparse
from collections import defaultdict
import cProfile
import functools
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab'), str(ROOT / 'GLIP')]
os.chdir(ROOT)
os.environ.setdefault('NLTK_DATA', str(ROOT / 'data/nltk_data'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--indices', default='0,60,130')
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--tf32', action='store_true')
    args = parser.parse_args()
    import habitat
    import numpy as np
    import torch
    import utils.instancenav_dataset
    from PSG_Nav import PSG_Nav_Agent
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    config = habitat.get_config('configs/instancenav_val20_qwen.yaml')
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    agent_args = SimpleNamespace(dataset='instancenav', split_l=-1, split_r=-1, server_port=None,
        visualize=False, timestamp='profile_' + args.name, rag_mode=3, rag_max_docs=10,
        rag_similarity_threshold=.1, rag_caption_penalty=1., rag_window_size=5)
    stats = defaultdict(lambda: {'calls': 0, 'seconds': 0.})

    def wrap(obj, method, label):
        original = getattr(obj, method)
        @functools.wraps(original)
        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stats[label]['seconds'] += time.perf_counter() - start
                stats[label]['calls'] += 1
        setattr(obj, method, timed)

    output = ROOT / f'data/downloads/profile_{args.name}.json'
    records = []
    with habitat.Env(config=config) as env:
        env.seed(config.SEED)
        agent = PSG_Nav_Agent(config, agent_args)
        agent.simulator = SimpleNamespace(_env=env)
        for method in ('update_map', 'update_free_map', 'update_room_map', 'detect_objects'):
            wrap(agent, method, method)
        for method in ('segment2d', 'mapping3d', 'update_node', 'update_edge', 'update_group', 'perception'):
            wrap(agent.scenegraph, method, 'graph.' + method)
        wrap(agent.glip_demo, 'inference', 'glip.inference')
        wrap(agent.scenegraph.model_client, 'complete', 'api.complete')
        validation = json.loads((ROOT / 'data/downloads/scenes_validation.json').read_text())
        episodes = {(ep.episode_id, ep.scene_id): ep for ep in env.episodes}
        profiler = cProfile.Profile()
        for index in map(int, args.indices.split(',')):
            entry = validation[index]
            env.current_episode = episodes[(entry['episode_id'], entry['scene_id'])]
            obs = env.reset()
            agent.reset()
            profiler.enable()
            for step in range(args.steps):
                started = time.perf_counter()
                action = agent.act(obs)
                obs = env.step(action)
                agent.update_metrics(env.get_metrics())
                torch.cuda.synchronize()
                records.append({'index': index, 'episode_id': entry['episode_id'], 'step': step + 1,
                    'seconds': time.perf_counter() - started, 'action': action,
                    'episode_over': env.episode_over, 'metrics': dict(env.get_metrics())})
                if (step + 1) % 10 == 0 or env.episode_over:
                    snapshot = {'name': args.name, 'torch_threads': torch.get_num_threads(), 'tf32': args.tf32,
                        'steps': records, 'timings_inclusive': dict(stats),
                        'api': dict(agent.scenegraph.model_client.stats)}
                    output.write_text(json.dumps(snapshot, indent=2))
                    print('PROFILE_PROGRESS', index, step+1, 'mean_s', round(np.mean([r['seconds'] for r in records]), 3), flush=True)
                if env.episode_over:
                    break
            profiler.disable()
        output.write_text(json.dumps({'name': args.name, 'torch_threads': torch.get_num_threads(), 'tf32': args.tf32,
            'steps': records, 'timings_inclusive': dict(stats),
            'api': dict(agent.scenegraph.model_client.stats)}, indent=2))
        profiler.dump_stats(str(output.with_suffix('.pstats')))
        print('PROFILE_COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
