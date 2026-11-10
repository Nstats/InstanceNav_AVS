#!/usr/bin/env python3
"""Export frozen S6 calibration and verify streaming feature/q/rule parity. No API calls."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_yes_pool import load
from tools import analyze_relative_evidence as R
from tools.build_verification_set import world_to_gps
from utils.relative_evidence import CandidatePool, Calibration, FEATURES, decision_snapshot


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fit(M, tasks, params):
    rows = [m for i in tasks for m in M[i] if m['t'] % 50 == 0]
    mu, scale, weights = R.fit_logit(np.vstack([m['X'] for m in rows]),
                                    np.concatenate([(m['tdist'] < 1.0).astype(float) for m in rows]))
    return {'mean': mu.tolist(), 'scale': scale.tolist(), 'coef': weights[:-1].tolist(),
            'intercept': float(weights[-1]), 'params': params,
            'training_tasks': len(tasks), 'training_rows': sum(len(m['X']) for m in rows)}


def export(output, report):
    eps, M, END, scenes_by_task, _ = load()
    tasks = sorted(M)
    scenes = sorted(set(scenes_by_task.values()))
    np.random.default_rng(20261104).shuffle(scenes)
    folds = {scene: index % 5 for index, scene in enumerate(scenes)}
    params_path = ROOT / 'outputs/analysis_relative_evidence_20261104/ablation_params.json'
    selected = json.loads(params_path.read_text())['p1_drop_rej_Ucount']
    models = []
    for fold in range(5):
        p = selected[fold]
        models.append(fit(M, [i for i in tasks if folds[scenes_by_task[i]] != fold],
                          {'alpha': p['a'], 'c': p['c'], 'tau': p['tau']}))
        print(f'Fitted fold {fold}', flush=True)
    source_paths = ['tools/analyze_relative_evidence.py', 'tools/analyze_yes_pool.py',
                    'outputs/analysis_relative_evidence_20261104/ablation_params.json',
                    'data/datasets/instancenav/val10/val10.json.gz', 'data/verification_set/manifest.json']
    data = {'schema_version': 1, 'features': FEATURES, 'fallback_policy_step': 860,
            'scene_folds': folds, 'folds': models,
            'global': fit(M, tasks, {'alpha': .1, 'c': 20, 'tau': .3}),
            'source': {'run': R.RUN, 'exploration_description': True,
                       'model': 'qwen3.5-flash', 'n_episodes': len(tasks), 'fold_seed': 20261104,
                       'fit': {'sample_policy_step_interval': 50, 'l2': 1., 'iterations': 500},
                       'files_sha256': {p: digest(ROOT / p) for p in source_paths}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    calibration = Calibration(output)
    ds = json.load(gzip.open(ROOT / 'data/datasets/instancenav/val10/val10.json.gz', 'rt'))['episodes']
    ds = {(e['scene_id'].split('/')[-1], str(e['episode_id']), str(e['goal_object_id'])): e for e in ds}
    moments, boxes, max_q_error = 0, 0, 0.
    for i in tasks:
        ep = eps[i]
        episode = ds[tuple(ep['key'])]
        positions = {f['step']: world_to_gps(f['position'], episode['start_position'], episode['start_rotation'])
                     for f in ep['frames']}
        by_step = defaultdict(list)
        for index, b in enumerate(ep['boxes']):
            if b.get('step') is not None and b.get('est_gps'):
                by_step[b['step']].append((index, b))
        pool = CandidatePool()
        reference_pool = R.Pool()
        model, _ = calibration.for_scene(ep['key'][0])
        moment_by_time = {m['t']: m for m in M[i]}
        position = None
        for step in sorted(set(by_step) | set(moment_by_time)):
            for index, b in by_step.get(step, []):
                agent_pos = positions.get(step)
                distance = math.dist(agent_pos, b['est_gps']) if agent_pos is not None else b.get('est_distance_m')
                pool.add(event_id=index, step=step, gps=b['est_gps'],
                         answer=b.get('confirmation') or b.get('admission'),
                         agent_position=agent_pos, observed_distance=distance)
                reference_pool.add({**b, 'agent_dist': distance})
                assert pool.candidates[-1].seed == reference_pool.c[-1]['seed']
                boxes += 1
            if step in positions:
                position = positions[step]
            if step not in moment_by_time:
                continue
            m = moment_by_time[step]
            x = np.array([c.features() for c in pool.candidates])
            np.testing.assert_allclose(x, m['X'], rtol=0, atol=1e-12)
            np.testing.assert_allclose([c.position for c in pool.candidates],
                                       [R.cand_pos(c) for c in reference_pool.c], rtol=0, atol=1e-12)
            q = Calibration.predict(model, x)
            ref_q = R.predict((np.array(model['mean']), np.array(model['scale']),
                               np.r_[model['coef'], model['intercept']]), m['X'])
            max_q_error = max(max_q_error, float(np.max(np.abs(q - ref_q))))
            np.testing.assert_allclose(q, ref_q, rtol=0, atol=1e-12)
            snapshot = decision_snapshot(pool, model, step=step, position=position, remaining=m['remaining'])
            alive = ~m['no']
            lead = int(np.flatnonzero(alive)[np.argmax(ref_q[alive])]) if alive.any() else None
            assert snapshot['leader'] == lead
            assert snapshot['reachable'] == R.reachable(m).tolist()
            order = sorted(np.flatnonzero(m['yes'] & R.reachable(m)), key=lambda k: (-ref_q[k], k))
            assert snapshot['ranked_yes'] == order
            p = model['params']
            expected_p1 = (ref_q[lead] / (ref_q[alive].sum() + p['alpha'] * max(0, p['c'] - m['n_seen']))
                           if lead is not None else 0.)
            assert abs(snapshot['p1'] - expected_p1) < 1e-12
            expected_trigger = ('fallback' if step >= 860 else 'relative_threshold'
                                if lead is not None and m['yes'][lead] and R.reachable(m)[lead]
                                and expected_p1 >= p['tau'] else None)
            assert snapshot['trigger'] == expected_trigger
            moments += 1
    result = {'episodes': len(tasks), 'observations': boxes, 'moments': moments,
              'max_q_absolute_error': max_q_error, 'calibration_sha256': calibration.sha256,
              'checks': 'seed clustering, candidate positions, all features, q, reachability, leader, ranking, p1, triggers',
              'passed': True}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/relative_evidence_s6.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/relative_evidence_parity.json')
    args = parser.parse_args()
    export(args.output, args.report)
