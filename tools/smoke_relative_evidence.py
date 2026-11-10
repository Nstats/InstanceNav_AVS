#!/usr/bin/env python3
"""Run exactly three fixed task identities under baseline/method, with category-only search."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab'), str(ROOT / 'GLIP')]
CONFIG = 'configs/runtime/instancenav_unseen10_1000atomic_1m.yaml'
SOURCE_INDICES = (17, 2, 11)


def task_mapping():
    import habitat
    from utils.instancenav_dataset import InstanceNavDataset
    config = habitat.get_config(str(ROOT / CONFIG))
    dataset = InstanceNavDataset(config.DATASET)
    ordered = list(dataset.get_episode_iterator(seed=config.SEED, **{
        k.lower(): v for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()}))
    def key(e):
        return (Path(e.scene_id).name, str(e.episode_id), str(e.goal_object_id))
    lookup = {key(e): i for i, e in enumerate(ordered)}
    raw = json.load(gzip.open(ROOT / config.DATASET.DATA_PATH, 'rt'))['episodes']
    return [{'source_index': i, 'runtime_index': lookup[(Path(raw[i]['scene_id']).name,
              str(raw[i]['episode_id']), str(raw[i]['goal_object_id']))],
             'category': raw[i]['object_category'], 'episode_id': str(raw[i]['episode_id']),
             'scene': Path(raw[i]['scene_id']).name, 'goal_object_id': str(raw[i]['goal_object_id'])}
            for i in SOURCE_INDICES]


def runtime_sources():
    paths = [ROOT / 'PSG_Nav.py', ROOT / 'habitat-lab/habitat/core/benchmark.py']
    for directory in ('utils', 'configs', 'graph', 'scripts'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.json', '.sh'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment(mode, gpu):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': ':'.join(str(ROOT / p) for p in ('habitat-lab', '.', 'GLIP')),
           'PSGNAV_ENV_FILE': str(ROOT / '.env'), 'PSGNAV_MODEL_BACKEND': 'openai',
           'QWEN_MODEL_NAME': 'qwen3.5-flash', 'PSGNAV_API_CONCURRENCY': '8',
           'PSGNAV_API_MAX_TOKENS': '4096', 'PSGNAV_API_CACHE_SIZE': '4096',
           'PSGNAV_EXACT_CACHE': '1', 'PSGNAV_TF32': '1', 'PSGNAV_TORCH_THREADS': '4',
           'HF_HOME': str(ROOT / 'data/huggingface'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'NLTK_DATA': str(ROOT / 'data/nltk_data'), 'LANDMARK_COMPARISON_REPEATS': '1',
           'LANDMARK_COMPARISON_WORKERS': '10', 'ENABLE_OBJECT_RELATIONS': 'false',
           'ENABLE_GROUP_LLM_FILTER': 'true', 'ENABLE_ROOM_LLM_FILTER': 'true',
           'ENABLE_INSTANCE_CONFIRMATION': 'true', 'INSTANCE_CONFIRMATION_THRESHOLD': '1',
           'VERIFY_BEFORE_CANDIDATE': 'false', 'INSTANCE_CONFIRMATION_MODE': 'positive_match',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false',
           'RAG_MODE': '0', 'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'false',
           'ENABLE_RELATIVE_EVIDENCE': 'true' if mode == 'method' else 'false',
           'RELATIVE_EVIDENCE_CALIBRATION': str(ROOT / 'configs/relative_evidence_s6.json'),
           'RELATIVE_EVIDENCE_USE_U': 'true', 'RELATIVE_EVIDENCE_EARLY_STOP': 'true',
           'RELATIVE_EVIDENCE_PAIRWISE': 'true', 'RELATIVE_EVIDENCE_DROP_REJECTED': 'true'}
    env['HUGGINGFACE_HUB_CACHE'] = env['TRANSFORMERS_CACHE'] = str(ROOT / 'data/huggingface/hub')
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    env.pop('PSGNAV_PROGRESS_FD', None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('run-id must contain only letters, digits, hyphens and underscores')
    gpus = [int(v) for v in args.gpus.split(',')]
    if len(gpus) != 2 or gpus[0] == gpus[1]:
        raise ValueError('Specify two distinct GPU ids')
    os.chdir(ROOT)
    output = ROOT / 'outputs' / args.run_id
    output.mkdir(exist_ok=False)
    tasks, sources = task_mapping(), runtime_sources()
    (output / 'manifest.json').write_text(json.dumps({'tasks': tasks, 'sources': sources,
        'settings': 'category search; same qwen3.5-flash, seed 77, 1000 atomic actions'}, indent=2))

    def run_mode(mode, gpu):
        env = environment(mode, gpu)
        rows = []
        for task in tasks:
            if runtime_sources() != sources:
                raise RuntimeError('Runtime sources changed during smoke evaluation')
            stamp = f'{args.run_id}_{mode}_ep{task["source_index"]}'
            log = output / f'{mode}_ep{task["source_index"]}.log'
            command = [sys.executable, '-u', 'PSG_Nav.py', '--dataset', 'instancenav',
                       '--dataset_split', 'val', '--config', CONFIG, '--rag_mode', '0',
                       '--episode_start', str(task['runtime_index']),
                       '--episode_end', str(task['runtime_index'] + 1), '--timestamp', stamp]
            env['PSGNAV_LOG_FILE'] = str(log)
            print(f'START {mode} {task}', flush=True)
            started = time.monotonic()
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                raise RuntimeError(f'{mode} task {task["source_index"]} failed: {log}')
            records = (ROOT / f'data/results_instancenav_{stamp}/experiment_0/results.txt').read_text().splitlines()
            if len(records) != 1:
                raise RuntimeError('Smoke must evaluate exactly one episode per invocation')
            metrics = ast.literal_eval(records[0])
            assert str(metrics['episode_id']) == task['episode_id']
            assert Path(metrics['scene_id']).name == task['scene']
            assert str(metrics['goal_object_id']) == task['goal_object_id']
            assert metrics['num_steps'] <= 1000
            row = {'mode': mode, **task, 'metrics': metrics, 'seconds': time.monotonic() - started,
                   'log': str(log.relative_to(ROOT)), 'timestamp': stamp}
            rows.append(row)
            (output / f'{mode}.json').write_text(json.dumps(rows, indent=2))
            print(f'DONE {mode} source_index={task["source_index"]} {metrics}', flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_mode, mode, gpu) for mode, gpu in zip(('baseline', 'method'), gpus)]
        rows = [row for future in futures for row in future.result()]
    if runtime_sources() != sources:
        raise RuntimeError('Runtime sources changed during smoke evaluation')
    (output / 'summary.json').write_text(json.dumps({'completed': len(rows), 'rows': rows,
                                                   'runtime_sources_unchanged': True}, indent=2))
    print(f'Smoke complete: {output}', flush=True)


if __name__ == '__main__':
    main()
