#!/usr/bin/env python3
"""Audit collection completeness without models, simulator replay or ground truth.

Writes collection_check.json. A nonzero exit means the run must not be treated as
a complete collection. Incomplete earlier attempts are counted but not mixed into
the final completed attempts. Run automatically by scripts/run_coin.sh.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_coin import load_rows


def check(root):
    root = Path(root)
    info = json.loads((root / 'run_info.json').read_text())
    rows = load_rows(root)
    errors = []
    if info.get('evaluation_kind') != 'collection' or info.get('collection_schema_version') != 2:
        errors.append('Expected collection schema version 2')
    if len(rows) != info.get('total'):
        errors.append(f'Only {len(rows)} of {info.get("total")} episodes completed')
    if any(r.get('num_steps') != 500 for r in rows):
        errors.append('Some result rows did not complete exactly 500 atomic actions')
    episodes = defaultdict(list)
    directories = sorted((ROOT / 'outputs').glob(f'coin_{info["timestamp"]}_gpu*.collect'))
    if not directories:
        errors.append('Missing .collect directories')
    for directory in directories:
        event_file = directory / 'events.jsonl'
        if not event_file.exists():
            errors.append(f'Missing {event_file}')
            continue
        for line_number, line in enumerate(event_file.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                episodes[event['episode_attempt_id']].append((directory, event))
            except (ValueError, KeyError):
                errors.append(f'Malformed event: {event_file}:{line_number}')
    totals = Counter()
    completed = {}
    for attempt, items in episodes.items():
        ends = [e for _, e in items if e['tag'] == 'Collect End']
        if not ends:
            totals['incomplete_attempts'] += 1
            continue
        end = ends[-1]
        key = (Path(end['scene_id']).name, end['episode_id'])
        if key in completed:
            errors.append(f'Duplicate completed episode {key}')
        completed[key] = end
        if len(ends) != 1 or end['atomic_steps'] != 500 or end['budget'] != 500 or end['found_goal']:
            errors.append(f'Invalid collection termination: {key}')
        frames, decisions, results, requests = {}, {}, {}, set()
        for directory, event in items:
            tag = event['tag']
            totals[tag] += 1
            if tag == 'Collect Observation':
                frames[event['frame_id']] = event
                assets = [event.get('rgb'), event.get('depth')]
                if not all(assets):
                    errors.append(f'Missing RGB/depth metadata: {event["frame_id"]}')
                for candidate in event['candidates']:
                    if candidate.get('collect_track_id') is None:
                        errors.append('Candidate without passive track ID')
                    if candidate['detector_score'] is None:
                        totals['missing_detector_score'] += 1
            else:
                assets = []
            if tag == 'Collect Decisions':
                decisions[event['frame_id']] = event
                for candidate in event['candidates']:
                    for answer in candidate['decisions'].values():
                        totals[answer] += 1
                    if not candidate['decisions']:
                        totals['not_queried'] += 1
            if tag == 'Instance Verification':
                request_key = (event['verification_id'], event.get('attempt'))
                if event['event'] == 'request':
                    requests.add(request_key)
                    assets.append(event['image'])
                elif event['event'] == 'result':
                    results[event['verification_id']] = event
                    if request_key not in requests:
                        errors.append(f'Result without saved request: {request_key}')
                elif event['event'] == 'raw_response':
                    if not isinstance(event.get('cache_hit'), bool) or not event.get('cache_key'):
                        errors.append(f'Missing cache metadata: {request_key}')
                    totals['cache_hits'] += event.get('cache_hit') is True
            for asset in assets:
                if asset and not (directory / asset['path']).is_file():
                    errors.append(f'Missing asset {directory / asset["path"]}')
        if set(frames) != set(decisions):
            errors.append(f'Observation/decision mismatch: {key}')
        for frame_id, decision in decisions.items():
            observation = frames.get(frame_id)
            if observation is None:
                continue
            observed = [(c['box_index'], c['collect_track_id']) for c in observation['candidates']]
            decided = [(c['box_index'], c['collect_track_id']) for c in decision['candidates']]
            if observed != decided:
                errors.append(f'Candidate alignment mismatch: {frame_id}')
            for stage in decision['stages']:
                result = results.get(stage['verification_id'])
                if (result is None or result['matches'] != stage['matches']
                        or result['box_indices'] != stage['box_indices']
                        or len(stage['matches']) != len(stage['box_indices'])):
                    errors.append(f'Stage/result mismatch: {frame_id}')
    expected = {(Path(r['scene_id']).name, str(r['episode_id'])) for r in rows}
    if set(completed) != expected:
        errors.append('Completed collection episodes do not match results.txt')
    # The environment log is authoritative for actions, including STOP at step 500.
    action_counts = Counter()
    last_atomic = {}
    for directory in directories:
        log = directory.with_suffix('.log')
        if not log.exists():
            errors.append(f'Missing {log}')
            continue
        active_attempt = None
        for line in log.read_text().splitlines():
            if line.startswith('[Collect Episode] '):
                active_attempt = json.loads(line[len('[Collect Episode] '):])['episode_attempt_id']
            if line.startswith('[Step] '):
                step = json.loads(line[len('[Step] '):])
                if step['action'] == 0:
                    errors.append(f'Actual STOP in {log.name} at policy step {step["policy_step"]}')
                totals['policy_steps'] += 1
                action_counts[active_attempt] += step.get('primitives', 0)
                last_atomic[active_attempt] = step.get('atomic_steps')
    for end in completed.values():
        attempt = end['episode_attempt_id']
        if action_counts[attempt] != 500 or last_atomic.get(attempt) != 500:
            errors.append(f'Action log does not cover exactly 500 atomic actions: {attempt}')
    report = {'ok': not errors, 'episodes': len(completed), 'expected_episodes': info.get('total'),
              'counts': dict(totals), 'errors': errors,
              'note': 'Collection completeness only; no navigation SR or offline rule score.'}
    (root / 'collection_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root')
    args = parser.parse_args()
    report = check(args.results_root)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
