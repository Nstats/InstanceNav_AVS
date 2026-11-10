#!/usr/bin/env python3
"""Compare execution settings on identical real HM3D frames, including output drift."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'habitat-lab'), str(ROOT/'GLIP')]
os.chdir(ROOT)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PSGNAV_EXACT_CACHE'] = '0'
os.environ['NLTK_DATA'] = str(ROOT/'data/nltk_data')


def main():
    import habitat
    import numpy as np
    import torch
    import utils.instancenav_dataset
    from PSG_Nav import PSG_Nav_Agent
    from utils.inference_cache import install_exact_caches
    from utils.utils_glip import categories_21
    config = habitat.get_config('configs/instancenav_val20_qwen.yaml')
    agent_args = SimpleNamespace(dataset='instancenav', split_l=-1, split_r=-1, server_port=None,
        visualize=False, timestamp='perception_bench', rag_mode=3, rag_max_docs=10,
        rag_similarity_threshold=.1, rag_caption_penalty=1., rag_window_size=5)
    frames = []
    validation = json.loads((ROOT/'data/downloads/scenes_validation.json').read_text())
    with habitat.Env(config=config) as env:
        by_id = {(ep.episode_id, ep.scene_id): ep for ep in env.episodes}
        for index in (0, 60, 130):
            entry = validation[index]
            env.current_episode = by_id[(entry['episode_id'], entry['scene_id'])]
            frames.append(env.reset()['rgb'].copy())
    agent = PSG_Nav_Agent(config, agent_args)
    from utils.utils_glip import categories_21
    baseline = {}
    report = {}

    def run(frame):
        bgr = frame[:, :, [2, 1, 0]]
        timings = {}
        torch.cuda.synchronize()
        started = time.perf_counter()
        objects = agent.glip_demo.inference(bgr, categories_21)
        torch.cuda.synchronize()
        timings['objects'] = time.perf_counter() - started
        started = time.perf_counter()
        rooms = agent.glip_demo.inference(bgr, agent.rooms)
        torch.cuda.synchronize()
        timings['rooms'] = time.perf_counter() - started
        started = time.perf_counter()
        segmentation = agent.scenegraph.get_sam_segmentation_dense(
            'groundedsam', agent.scenegraph.mask_generator, frame)
        torch.cuda.synchronize()
        timings['segmentation'] = time.perf_counter() - started
        return {'objects': objects, 'rooms': rooms, 'segmentation': segmentation}, timings

    def compare(reference, candidate):
        result = {}
        for name in ('objects', 'rooms'):
            a, b = reference[name], candidate[name]
            equal = len(a) == len(b) and torch.equal(a.get_field('labels'), b.get_field('labels'))
            result[name] = {'count': len(b), 'labels_equal': equal,
                'bbox_max_abs': float((a.bbox-b.bbox).abs().max()) if equal and len(a) else None,
                'scores_max_abs': float((a.get_field('scores')-b.get_field('scores')).abs().max()) if equal and len(a) else None}
        a, b = reference['segmentation'], candidate['segmentation']
        same = a[3] == b[3]
        result['segmentation'] = {'labels_equal': same, 'count': len(b[3]) if b[3] is not None else 0}
        if same and a[0] is not None and a[0].shape == b[0].shape:
            result['segmentation']['mask_agreement'] = float(np.mean(a[0] == b[0]))
            result['segmentation']['boxes_max_abs'] = float(np.max(np.abs(a[1]-b[1])))
        return result

    for name, threads, tf32 in [('baseline20',20,False), ('threads4',4,False),
                                ('tf32_threads4',4,True), ('exact_cache_threads4',4,False)]:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        if name.startswith('exact_cache'):
            install_exact_caches(agent)
        rows = []
        for index, frame in enumerate(frames):
            for repeat in range(2):
                output, timings = run(frame)
                if name == 'baseline20' and repeat == 1:
                    baseline[index] = output
                row = {'frame': index, 'repeat': repeat, 'seconds': timings}
                if name != 'baseline20':
                    row['comparison'] = compare(baseline[index], output)
                rows.append(row)
                print('PERCEPTION_BENCH', name, index, repeat, json.dumps(row), flush=True)
        report[name] = rows
        (ROOT/'data/downloads/perception_benchmark.json').write_text(json.dumps(report, indent=2))
    print('PERCEPTION_BENCH_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
