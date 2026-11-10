#!/usr/bin/env python3
"""Summarize one CoIN result directory (data/results_coin_<timestamp>/).

Reports SR, SPL, SoftSPL, mean final DTG and the distractor-stop rate
(distractor_success), with a 95% bootstrap interval on SR that resamples
targets, not episodes: CoIN gives each target about ten starts with one
instruction, so episodes of one target are not independent. Writes
summary.json and per_episode.csv next to the results.

  python tools/summarize_coin.py data/results_coin_val_seen_20261106_120000_123
"""
import argparse
import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

METRICS = ('success', 'spl', 'softspl', 'distance_to_goal', 'distractor_success')


def load_rows(root):
    root = Path(root)
    merged = root / 'experiment_0' / 'results.txt'
    files = [merged] if merged.exists() else sorted(root.glob('gpu*/results.txt'))
    rows = []
    for path in files:
        rows += [ast.literal_eval(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(Path(str(r['scene_id'])).name, str(r['episode_id'])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{root}: duplicate (scene, episode_id) rows')
    return rows


def target_of(row):
    return Path(str(row['scene_id'])).name, str(row.get('goal_object_id'))


def bootstrap_sr(rows, reps=2000, seed=0):
    by_target = defaultdict(list)
    for row in rows:
        by_target[target_of(row)].append(float(row['success']))
    groups = list(by_target.values())
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(reps):
        sample = [rng.choice(groups) for _ in groups]
        rates.append(sum(map(sum, sample)) / sum(map(len, sample)))
    rates.sort()
    return [rates[int(0.025 * reps)], rates[int(0.975 * reps) - 1]]


def summarize(root):
    root = Path(root)
    rows = load_rows(root)
    info_path = root / 'run_info.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    summary = {'results_root': str(root), 'split': info.get('split'), 'subset': info.get('subset'),
               'expected_episodes': info.get('total'), 'episodes': len(rows),
               'targets': len({target_of(r) for r in rows})}
    for name in METRICS:
        values = [float(r[name]) for r in rows if r.get(name) is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
    interval = bootstrap_sr(rows)
    summary['success_ci95_target_bootstrap'] = None if interval is None else [round(v, 4) for v in interval]
    # Rows carry no termination field; an episode shorter than the 500-step budget ended by STOP.
    summary['ended_before_step_limit'] = sum(int(r.get('num_steps', 0)) < 500 for r in rows)
    collection = info.get('evaluation_kind') == 'collection' or info.get('switches', {}).get('INSTANCE_COLLECT_ONLY') == 'true'
    summary['evaluation_kind'] = 'collection' if collection else 'navigation'
    if collection:
        summary['raw_navigation_metrics_not_for_comparison'] = {k: summary[k] for k in METRICS}
        for name in METRICS:
            summary[name] = None
        summary['success_ci95_target_bootstrap'] = None
        summary['completed_budget_episodes'] = sum(int(r.get('num_steps', 0)) == 500 for r in rows)
    with open(root / 'per_episode.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'scene', 'episode_id', 'goal_object_id', 'goal', *METRICS, 'num_steps'])
        for row in rows:
            writer.writerow([summary['split'], Path(str(row['scene_id'])).name, row['episode_id'],
                             row.get('goal_object_id'), row.get('goal'),
                             *[row.get(name) for name in METRICS], row.get('num_steps')])
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results_roots', nargs='+')
    args = parser.parse_args()
    for root in args.results_roots:
        s = summarize(root)
        ci = s['success_ci95_target_bootstrap']
        missing = '' if s['expected_episodes'] in (None, s['episodes']) else f' (expected {s["expected_episodes"]})'
        if s['evaluation_kind'] == 'collection':
            print(f"{root}: COLLECTION episodes={s['episodes']}{missing} "
                  f"budget_500={s['completed_budget_episodes']} targets={s['targets']}; "
                  'SR/SPL not applicable (STOP disabled)')
            continue
        print(f"{s['split'] or root}: episodes={s['episodes']}{missing} targets={s['targets']} "
              f"SR={s['success']} CI95={ci} SPL={s['spl']} SoftSPL={s['softspl']} "
              f"DTG={s['distance_to_goal']} distractor_stop={s['distractor_success']} "
              f"ended_before_step_limit={s['ended_before_step_limit']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
