#!/usr/bin/env python
"""S6 close-out: replay-gate check and the "what could still be won" ceiling.

Two questions, one script:

  gate     Replay "stop at the first match" on the collect-only run (S6) and compare the
           result with the three real baselines (admin-off, k=1). The replay SR is the
           fraction of episodes whose FIRST intended lock sits on the target instance.
           This is the check that 实验计划.md S6 uses to decide whether the replay is a
           faithful stand-in for the closed loop. If it lands outside the baseline range,
           every S7 conclusion built on this trajectory is suspect.

           The same computation is run on the three real admin-off runs, where the first
           real lock is known: predicted SR (first lock on target) should equal the actual
           SR. That validates the criterion itself before it is applied to the collect run.

  ceiling  On episodes where "stop at the first match" fails, how much is actually still
           available afterwards? Three nested questions, all from ground truth, all read
           only for evaluation:
             seen     the target is ever >= VIS_PX pixels after the failed first lock
             near     the target is ever within NEAR_DISTANCE_M of the camera afterwards
             yes      the target is ever judged 'yes' afterwards (a stopping opportunity
                      the baseline rule would have taken had it not already stopped)
           'yes' is the practical ceiling for a stop-when-confident rule: a task can only
           be won by postponing if the target is eventually judged yes while the agent is
           still there.

Run from the repository root:
  PYTHONPATH=habitat-lab:. $PSGNAV_PYTHON tools/analyze_replay_first_match.py
Writes outputs/analysis_replay_first_match_<date>/ and a report next to it.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.analyze_verification_set import load_set  # noqa: E402

VIS_PX = 2000            # "seen" threshold used by F13/F14 and the S5 oracle
NEAR_DISTANCE_M = 4.0    # PSG_Nav_Agent.distance_threshold: near-range confirmation cut-off
COLLECT_GROUP = 'collect'
BASELINE_GROUPS = ('adm', 'def')
N_BOOT = 5000
RNG = np.random.default_rng(20261103)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def first_lock(ep, key):
    locks = ep.get(key) or []
    if not locks:
        return None
    return min(locks, key=lambda L: L.get('step') or 0)


def lock_on_target(ep, L):
    """Was the lock geometrically on the goal instance, i.e. within 1 m of it?

    Geometric, not label-based: boxes on a chair that stands next to the goal chair get
    labelled 'same_category' (the neighbour has more pixels in the box), yet stopping there
    is a success because success is distance to the goal instance. On the six baseline runs
    the label rule under-counts target locks by 13-16 episodes; this rule agrees with the
    actual outcome on 81-88% of episodes.
    """
    if L is None:
        return None
    g = L.get('goal_gps')
    tgt = (ep.get('same_category_instances') or {}).get(str(ep.get('target_instance')))
    if not g or not tgt or not tgt.get('gps'):
        return None
    return math.dist(g, tgt['gps']) <= 1.0


def box_answers_by_step(ep):
    """Per step: was a target box judged yes, and how close was the nearest target box."""
    yes_steps, near_steps = set(), set()
    for b in ep['boxes']:
        s = b.get('step')
        if s is None:
            continue
        is_target = b.get('label') == 'target'
        if not is_target:
            continue
        if (b.get('confirmation') == 'yes') or (b.get('admission') == 'yes'):
            yes_steps.add(s)
        d = b.get('top_depth_m')
        if d is None:
            d = b.get('est_distance_m')
        if d is not None and d <= NEAR_DISTANCE_M:
            near_steps.add(s)
    return yes_steps, near_steps


def seen_steps(ep, vis_px=VIS_PX):
    return {x['step'] for x in ep.get('exposure') or [] if x.get('target_px', 0) >= vis_px}


def after(steps, t):
    return sorted(s for s in steps if s > t)


def boot_ci(values, n=N_BOOT):
    """Mean and 95% task-bootstrap interval of a list of per-task 0/1 or fractional values."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None, None, None
    m = float(a.mean())
    idx = RNG.integers(0, a.size, size=(n, a.size))
    b = a[idx].mean(axis=1)
    return m, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------------------
# Gate: replay "stop at the first match"
# --------------------------------------------------------------------------------------
def replay_first_match(data, run, lock_key):
    rows = []
    for idx, ep in sorted(data[run]['episodes'].items()):
        L = first_lock(ep, lock_key)
        rows.append({'run': run, 'task': idx, 'scene': ep['key'][0],
                     'lock_step': L.get('step') if L else None,
                     'on_target': lock_on_target(ep, L),
                     'n_unlocks': len(ep.get('unlocks') or []),
                     'n_locks': len(ep.get(lock_key) or []),
                     'success': bool(ep.get('success'))})
    return rows


# Empirical P(success | outcome of the FIRST lock), fitted on the six baseline k=1 runs
# (181 target locks, 86 wrong locks later released by stuck recovery, 209 wrong locks never
# released, 124 episodes with no lock at all). It is what a replay needs to turn "the lock
# the baseline would have made" into an expected success rate: the closed loop is not
# "lock right => success" (0.89), because 3 episodes were still lost after a correct lock
# (stuck on the way) and a wrong lock is sometimes rescued (0.37 with an unlock, 0.22 not).
P_SUCCESS = {'on_target': 0.890, 'wrong_unlocked': 0.372, 'wrong_locked': 0.220, 'no_lock': 0.0}


def expected_success(rows, lock_key):
    vals = []
    for r in rows:
        if r['on_target'] is True:
            k = 'on_target'
        elif r['on_target'] is False:
            k = 'wrong_unlocked' if r['n_unlocks'] > 0 else 'wrong_locked'
        else:
            k = 'no_lock'
        vals.append(P_SUCCESS[k])
    return vals


def _real_lock_on_target(L):
    """For real runs the verification set already resolves the lock identity in load_episode."""
    if L is None:
        return None
    ident = L.get('identity')
    if ident is not None:
        return ident == 'target'
    return None


# --------------------------------------------------------------------------------------
# Ceiling: what is still winnable after the first failed stop
# --------------------------------------------------------------------------------------
def ceiling_rows(data, run):
    out = []
    for idx, ep in sorted(data[run]['episodes'].items()):
        L = first_lock(ep, 'intended_locks')
        on_target = lock_on_target(ep, L)
        if on_target is None and L is None:
            on_target = False        # no lock at all: the baseline never stopped
        row = {'run': run, 'task': idx, 'scene': ep['key'][0],
               'lock_step': L.get('step') if L else None,
               'first_lock_on_target': on_target,
               'success': bool(ep.get('success'))}
        if not on_target:
            t = (L.get('step') if L else None) or 0
            yes_steps, near_steps = box_answers_by_step(ep)
            sn = after(seen_steps(ep), t)
            nr = after(near_steps, t)
            ys = after(yes_steps, t)
            row.update({
                'n_locks': len(ep.get('intended_locks') or []),
                'seen_after': bool(sn),
                'seen_after_step': sn[0] if sn else None,
                'near_after': bool(nr),
                'near_after_step': nr[0] if nr else None,
                'yes_after': bool(ys),
                'yes_after_step': ys[0] if ys else None,
                'steps_left': max(0, 1000 - t),
            })
        else:
            row.update({'n_locks': len(ep.get('intended_locks') or []),
                        'seen_after': None, 'seen_after_step': None,
                        'near_after': None, 'near_after_step': None,
                        'yes_after': None, 'yes_after_step': None,
                        'steps_left': max(0, 1000 - ((L.get('step') if L else None) or 0))})
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='restrict to these collect runs')
    ap.add_argument('--root', default=str(REPO / 'data' / 'verification_set'))
    ap.add_argument('--results-root', default=str(REPO / 'data'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = load_set(Path(args.root), Path(args.results_root), runs=args.runs)
    by_group = defaultdict(list)
    for r in sorted(data):
        by_group[data[r]['group']].append(r)

    collect = by_group.get(COLLECT_GROUP, [])
    if not collect:
        sys.exit('no collect-only run in the verification set')
    baselines = [r for g in BASELINE_GROUPS for r in by_group.get(g, [])]

    # ---- gate on the real baselines: does the replay criterion reproduce SR? ----
    gate_real = []
    for r in baselines:
        rows = replay_first_match(data, r, 'locks')
        known = [x for x in rows if x['on_target'] is not None]
        pred = float(np.mean([x['on_target'] for x in known])) if known else None
        act = float(np.mean([x['success'] for x in rows])) if rows else None
        exp = float(np.mean(expected_success(rows, 'locks'))) if rows else None
        gate_real.append({'run': r, 'group': data[r]['group'], 'n': len(rows),
                          'n_with_lock': len(known), 'pred_sr': pred, 'expected_sr': exp,
                          'actual_sr': act,
                          'median_first_lock_step': _median([x['lock_step'] for x in known])})

    # ---- gate on the collect run ----
    gate_collect = []
    for r in collect:
        rows = replay_first_match(data, r, 'intended_locks')
        unknown = sum(x['on_target'] is None for x in rows)
        locked = [x for x in rows if x['on_target'] is not None]
        vals = [1.0 if x['on_target'] else 0.0 for x in locked]
        # Episodes with no intended lock count as failures for the replay:
        # the baseline would have run out of budget without a lock.
        vals_all = [1.0 if x['on_target'] else 0.0 for x in rows]
        m, lo, hi = boot_ci(vals_all)
        exp = expected_success(rows, 'intended_locks')
        em, elo, ehi = boot_ci(exp)
        gate_collect.append({
            'run': r, 'n': len(rows), 'n_locked': len(locked), 'n_unknown_label': unknown,
            'replay_sr': m, 'replay_sr_ci': [lo, hi],
            'expected_sr': em, 'expected_sr_ci': [elo, ehi],
            'replay_sr_locked_only': float(np.mean(vals)) if vals else None,
            'median_first_lock_step': _median([x['lock_step'] for x in locked]),
            'median_n_locks': _median([x['n_locks'] for x in rows]),
        })

    # ---- ceiling on the collect run ----
    ceil = []
    for r in collect:
        ceil += ceiling_rows(data, r)
    failed = [x for x in ceil if x['first_lock_on_target'] is False]
    ceiling = {'n_episodes': len(ceil), 'n_first_lock_failed': len(failed),
               'n_first_lock_on_target': sum(x['first_lock_on_target'] is True for x in ceil)}
    for key in ('seen_after', 'near_after', 'yes_after'):
        vals = [1.0 if x[key] else 0.0 for x in failed]
        m, lo, hi = boot_ci(vals)
        ceiling[key] = {'n': len(vals), 'frac': m, 'ci': [lo, hi]}
        ceiling[key + '_steps'] = _median([x[key + '_step'] for x in failed if x[key + '_step']])
    ceiling['steps_left_median'] = _median([x['steps_left'] for x in failed])

    # Nested: target seen AND judged yes afterwards.
    nested = [1.0 if (x['seen_after'] and x['yes_after']) else 0.0 for x in failed]
    m, lo, hi = boot_ci(nested)
    ceiling['seen_and_yes'] = {'n': len(nested), 'frac': m, 'ci': [lo, hi]}

    # ---- write ----
    outdir = Path(args.out or (REPO / 'outputs' / 'analysis_replay_first_match_20261103'))
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('gate_real.csv', gate_real), ('gate_collect.csv', gate_collect),
                       ('ceiling.csv', ceil)):
        with open(outdir / name, 'w', newline='') as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    res = {'runs': {r: data[r]['group'] for r in sorted(data)},
           'gate_real': gate_real, 'gate_collect': gate_collect, 'ceiling': ceiling,
           'vis_px': VIS_PX, 'near_distance_m': NEAR_DISTANCE_M}
    json.dump(res, open(outdir / 'results.json', 'w'), ensure_ascii=False, indent=1)

    print(json.dumps(res, ensure_ascii=False, indent=1))


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


if __name__ == '__main__':
    main()
