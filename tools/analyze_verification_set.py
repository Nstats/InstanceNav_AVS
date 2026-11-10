#!/usr/bin/env python
"""S3 of 实验计划.md: re-measure the motivation hypotheses H1-H4 on exact GT labels.

Input is the offline verification set built by tools/build_verification_set.py
(data/verification_set/<run>/ep_XXX.json). Ground truth is used for evaluation only.

Conventions (see the plan, S3 "统计要求"):
  * Rates are episode-macro: compute the rate per dataset episode (pooling the runs of one
    configuration), then average over episodes. Pooled (per-box) rates are reported beside
    them for comparison only, because a few episodes contribute hundreds of boxes.
  * Confidence intervals come from a cluster bootstrap over dataset episodes; the same
    episode in different runs of a configuration is one cluster.
  * "Visible" means the target (or a same-category instance) covers >= --vis-px pixels of
    the 640x480 frame seen at that policy step.
  * The object a lock was made on is the box whose est_gps equals the lock's goal_gps at
    the lock step (exact for all logged locks); its semantic label says what was locked.
  * The "vote" stage is goal_confirmation: in k=1 runs its first yes on a track is the lock.

Usage (repository root):
  python tools/analyze_verification_set.py \
      --out outputs/analysis_verification_set_20261102
Writes results.json, tables.md and fig_*.png to --out.
"""
import argparse
import ast
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FRAME_W, FRAME_H = 640, 480
FRAME_PX = FRAME_W * FRAME_H
AREA_BINS = [(0.0, 0.02, '<2%'), (0.02, 0.05, '2-5%'), (0.05, 0.20, '5-20%'), (0.20, 1.01, '>=20%')]
DEPTH_BINS = [(0.0, 1.5, '<1.5m'), (1.5, 2.5, '1.5-2.5m'), (2.5, 4.0, '2.5-4m'), (4.0, 1e9, '>=4m')]
INST_BINS = [(1, 2, '1'), (2, 4, '2-3'), (4, 8, '4-7'), (8, 1e9, '>=8')]
LABELS = ('target', 'same_category', 'other')
# Baseline / ablation groups that S3 reports on. Collect-only runs (S6) get their own group
# ('collect') but are deliberately NOT in GROUP_ORDER: they never stop, so the failure
# taxonomy and the stop analyses do not apply to them. Read them explicitly by group name.
GROUP_ORDER = ('off', 'def', 'k3', 'adm')
GROUP_NAME = {'off': '核验关闭', 'def': '默认（准入开，k=1）', 'k3': 'k=3（准入开）', 'adm': '准入关闭（基线，k=1）',
              'collect': '只采集不停（S6）'}
K1_GROUPS = ('def', 'adm')

TAXONOMY = [
    ('success', '成功'),
    ('ws_samecat_before', '目标可见前错停在同类物体上'),
    ('ws_samecat_after', '见过目标后仍错停在同类物体上'),
    ('ws_other', '错停在非同类物体上'),
    ('stop_far_target_lock', '锁对目标，但停在 1 m 外'),
    ('stop_no_lock', '未锁定就停止'),
    ('lim_never_visible', '耗尽步数：目标从未可见'),
    ('lim_not_boxed', '耗尽步数：目标可见但未被检出'),
    ('lim_not_accepted', '耗尽步数：目标进框但没得到锁定票'),
    ('lim_accepted_no_lock', '耗尽步数：目标得到过锁定票但未锁定'),
    ('lim_target_locked', '耗尽步数：锁定过目标但没到达'),
    ('lim_wrong_lock', '耗尽步数：锁定过错误物体'),
]


# --------------------------------------------------------------------------------------
# Statistics helpers (unit-tested in tests/test_analyze_verification_set.py)
# --------------------------------------------------------------------------------------
def bin_of(value, bins):
    """Name of the half-open bin [lo, hi) that holds value (None if outside or missing)."""
    if value is None:
        return None
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return None


def macro_rate(pairs):
    """pairs: {episode: (num, den)} -> mean over episodes with den > 0 of num/den."""
    vals = [n / d for n, d in pairs.values() if d > 0]
    return float(np.mean(vals)) if vals else None


def pooled_rate(pairs):
    num = sum(n for n, d in pairs.values())
    den = sum(d for n, d in pairs.values())
    return num / den if den else None


def cluster_bootstrap(pairs, stat, n_boot=2000, seed=0):
    """Percentile 95% CI of stat(pairs) resampling episode keys with replacement."""
    keys = [k for k, (n, d) in pairs.items() if d > 0]
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        sample = {}
        for j in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            sample[(k, j)] = pairs[k]
        v = stat(sample)
        if v is not None:
            out.append(v)
    if not out:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def rate_summary(pairs, n_boot=2000, seed=0):
    lo, hi = cluster_bootstrap(pairs, macro_rate, n_boot, seed)
    return {'macro': macro_rate(pairs), 'ci': [lo, hi], 'pooled': pooled_rate(pairs),
            'n_boxes': sum(d for _, d in pairs.values()), 'n_episodes': sum(1 for _, d in pairs.values() if d > 0)}


def paired_bootstrap(diffs, n_boot=2000, seed=0):
    """Mean of per-episode differences with a percentile CI (resampling episodes)."""
    diffs = list(diffs)
    if len(diffs) < 2:
        return None, None, None
    rng = random.Random(seed)
    means = sorted(float(np.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])) for _ in range(n_boot))
    return float(np.mean(diffs)), means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def kaplan_meier(times_events):
    """times_events: list of (time, event_bool). Returns sorted [(t, S(t))] at event times."""
    data = sorted(times_events, key=lambda x: (x[0], not x[1]))
    at_risk = len(data)
    surv, curve, i = 1.0, [], 0
    while i < len(data):
        t = data[i][0]
        events = censored = 0
        while i < len(data) and data[i][0] == t:
            events += data[i][1]
            censored += not data[i][1]
            i += 1
        if events and at_risk:
            surv *= 1.0 - events / at_risk
            curve.append((t, surv))
        at_risk -= events + censored
    return curve


def km_survival_at(curve, t):
    s = 1.0
    for ti, si in curve:
        if ti <= t:
            s = si
        else:
            break
    return s


def lag1_stats(sequences):
    """sequences: list of lists of 0/1 (one per instance, in time order).

    Returns (P(yes | previous yes), P(yes | previous no), n_prev_yes, n_prev_no).
    """
    yy = yn = ny = nn = 0
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            if a and b:
                yy += 1
            elif a and not b:
                yn += 1
            elif not a and b:
                ny += 1
            else:
                nn += 1
    p_after_yes = yy / (yy + yn) if yy + yn else None
    p_after_no = ny / (ny + nn) if ny + nn else None
    return p_after_yes, p_after_no, yy + yn, ny + nn


def lag1_shuffle_test(sequences, n_perm=1000, seed=0):
    """Within-instance shuffle null for the lag-1 difference P(y|y) - P(y|n).

    Shuffling inside each instance keeps every instance's own yes rate (heterogeneity),
    so an observed difference above the null means bursts in time, not just hard instances.
    """
    seqs = [list(s) for s in sequences if len(s) >= 2 and 0 < sum(s) < len(s)]
    obs_y, obs_n, _, _ = lag1_stats(seqs)
    if obs_y is None or obs_n is None:
        return None
    obs = obs_y - obs_n
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        sh = []
        for s in seqs:
            s2 = s[:]
            rng.shuffle(s2)
            sh.append(s2)
        a, b, _, _ = lag1_stats(sh)
        null.append(a - b)
    null.sort()
    p = (1 + sum(v >= obs for v in null)) / (1 + len(null))
    return {'observed_diff': obs, 'null_mean': float(np.mean(null)), 'null_95': null[int(0.95 * len(null))], 'p': p,
            'n_sequences': len(seqs)}


def first_yes_target_prob(seq):
    """seq: list of (is_target, p_yes). P(first yes lands on a target view | a yes occurs in seq)."""
    alive, p_t, p_any = 1.0, 0.0, 0.0
    for is_t, p in seq:
        hit = alive * p
        p_any += hit
        if is_t:
            p_t += hit
        alive *= 1.0 - p
    return p_t / p_any if p_any > 0 else None


def binom_tail(n, p, k):
    """P(Binomial(n, p) >= k)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def group_of(toggles):
    # Collect-only runs (S6) never stop and must not be pooled with def/adm: their episodes
    # run to the step limit, so their snapshots/answers are not comparable to the baseline.
    category_search = str(toggles.get('ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION', 'true')).lower() in ('false', '0', 'no')
    if str(toggles.get('ENABLE_RELATIVE_EVIDENCE')).lower() == 'true':
        return 'relative_category' if category_search else 'relative'
    if category_search:
        # Never silently pool the new search ablation into historical S3 groups.
        original = {**toggles, 'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'true'}
        return group_of(original) + '_category'
    if str(toggles.get('INSTANCE_COLLECT_ONLY')).lower() == 'true':
        return 'collect'
    if str(toggles.get('ENABLE_INSTANCE_CONFIRMATION')).lower() != 'true':
        return 'off'
    if str(toggles.get('INSTANCE_CONFIRMATION_THRESHOLD')) == '3':
        return 'k3'
    if str(toggles.get('VERIFY_BEFORE_CANDIDATE')).lower() == 'false':
        return 'adm'
    return 'def'


def is_truncated(box, margin=2):
    x0, y0, x1, y1 = box
    return x0 <= margin or y0 <= margin or x1 >= FRAME_W - 1 - margin or y1 >= FRAME_H - 1 - margin


def lock_box(ep, lock):
    """The box the lock was made on: same step, est_gps equal to goal_gps (prefer a yes)."""
    g = lock.get('goal_gps')
    if not g:
        return None
    cands = [b for b in ep['boxes'] if b.get('step') == lock.get('step') and b.get('est_gps')
             and math.dist(b['est_gps'], g) < 1e-3]
    if not cands:
        return None
    cands.sort(key=lambda b: ('yes' not in (b.get('confirmation'), b.get('admission')), b.get('box_index') or 0))
    return cands[0]


def lock_identity(ep, lock):
    """('target' | 'same_category' | 'other', gt_instance, source) for one lock."""
    b = lock_box(ep, lock)
    if b is not None and b.get('label') in LABELS:
        return b['label'], b.get('gt_instance'), 'box'
    d = lock.get('nearest_instance_dist_m')
    if d is not None and d <= 1.5:
        return ('target' if lock.get('nearest_is_target') else 'same_category'), lock.get('nearest_instance'), 'nearest'
    return 'other', None, 'nearest'


def load_episode(path, dtg=None):
    e = json.load(open(path))
    e['dtg'] = dtg
    for L in e['locks']:
        L['identity'], L['identity_instance'], L['identity_source'] = lock_identity(e, L)
    e['locks'].sort(key=lambda L: L['step'])
    return e


def load_results(results_root, run):
    path = Path(results_root) / f'results_instancenav_{run}' / 'experiment_0' / 'results.txt'
    out = {}
    if path.exists():
        for line in open(path):
            x = ast.literal_eval(line)
            out[int(x['dataset_index'])] = x
    return out


def load_set(root, results_root, runs=None):
    manifest = json.load(open(Path(root) / 'manifest.json'))
    data = {}
    for run, info in manifest['runs'].items():
        if runs and run not in runs:
            continue
        res = load_results(results_root, run)
        eps = {}
        for p in sorted((Path(root) / run).glob('ep_*.json')):
            idx = int(p.stem.split('_')[1])
            eps[idx] = load_episode(p, (res.get(idx) or {}).get('distance_to_goal'))
        data[run] = {'group': group_of(info['toggles']), 'episodes': eps}
    return data


# --------------------------------------------------------------------------------------
# Per-episode derived quantities
# --------------------------------------------------------------------------------------
def first_visible_step(ep, vis_px):
    for x in ep['exposure']:
        if x['target_px'] >= vis_px:
            return x['step']
    return None


def first_distractor_visible_step(ep, vis_px):
    tgt = str(ep['target_instance'])
    for x in ep['exposure']:
        if any(k != tgt and v >= vis_px for k, v in x['same_category_px'].items()):
            return x['step']
    return None


def n_scene_instances(ep):
    return len(ep['same_category_instances'])


def final_lock(ep):
    return ep['locks'][-1] if ep['locks'] else None


def classify(ep, vis_px):
    if ep['success']:
        return 'success'
    fv = first_visible_step(ep, vis_px)
    if ep['termination'] == 'stop_called':
        L = final_lock(ep)
        if L is None:
            return 'stop_no_lock'
        if L['identity'] == 'target':
            return 'stop_far_target_lock'
        if L['identity'] == 'same_category':
            return 'ws_samecat_after' if (fv is not None and fv <= L['step']) else 'ws_samecat_before'
        return 'ws_other'
    if any(L['identity'] == 'target' for L in ep['locks']):
        return 'lim_target_locked'
    if ep['locks']:
        return 'lim_wrong_lock'
    if fv is None:
        return 'lim_never_visible'
    tboxes = [b for b in ep['boxes'] if b.get('label') == 'target']
    if not tboxes:
        return 'lim_not_boxed'
    if not any(b.get('confirmation') == 'yes' for b in tboxes):
        return 'lim_not_accepted'
    return 'lim_accepted_no_lock'


# --------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------
def group_runs(data, group):
    return [r for r in sorted(data) if data[r]['group'] == group]


def analysis_taxonomy(data, vis_px):
    out = {}
    for g in GROUP_ORDER:
        runs = group_runs(data, g)
        if not runs:
            continue
        per_run = []
        for r in runs:
            per_run.append(Counter(classify(e, vis_px) for e in data[r]['episodes'].values()))
        keys = [k for k, _ in TAXONOMY]
        out[g] = {'runs': runs,
                  'per_run': [{k: c.get(k, 0) for k in keys} for c in per_run],
                  'mean': {k: float(np.mean([c.get(k, 0) for c in per_run])) for k in keys},
                  'sd': {k: float(np.std([c.get(k, 0) for c in per_run], ddof=1)) if len(per_run) > 1 else 0.0 for k in keys}}
    # Never-solved episodes over the six k=1 runs.
    k1 = [r for g in K1_GROUPS for r in group_runs(data, g)]
    succ = Counter()
    per_ep = defaultdict(list)
    for r in k1:
        for i, e in data[r]['episodes'].items():
            c = classify(e, vis_px)
            succ[i] += c == 'success'
            per_ep[i].append(c)
    never = sorted(i for i in per_ep if succ[i] == 0)
    fail_runs = Counter(c for i in never for c in per_ep[i])
    modal = Counter(Counter(per_ep[i]).most_common(1)[0][0] for i in never)
    cats = Counter(data[k1[0]]['episodes'][i]['category'] for i in never)
    out['never_solved_k1'] = {'n_runs': len(k1), 'episodes': never, 'failure_runs': dict(fail_runs),
                              'modal_class': dict(modal), 'by_category': dict(cats),
                              'always': sum(1 for i in per_ep if succ[i] == len(k1)),
                              'flaky': sum(1 for i in per_ep if 0 < succ[i] < len(k1))}
    return out


def analysis_wrong_locks(data, vis_px, n_boot):
    """H1: hard distractors, timing, repeat identity, distance, order luck; item 6 non-category stops."""
    out = {}
    k1 = [r for g in K1_GROUPS for r in group_runs(data, g)]
    # Lock timing by identity (first lock of each run-episode, k=1 runs).
    timing = defaultdict(list)
    for r in k1:
        for e in data[r]['episodes'].values():
            if e['locks']:
                L = e['locks'][0]
                timing[L['identity']].append(L['step'])
    out['first_lock_step_median'] = {k: float(np.median(v)) for k, v in timing.items()}
    out['first_lock_n'] = {k: len(v) for k, v in timing.items()}
    # Same-category wrong stops: repeated identity across the six runs, distance to target.
    by_ep = defaultdict(list)
    dists, seen_flags = [], []
    for r in k1:
        for i, e in data[r]['episodes'].items():
            if classify(e, vis_px) in ('ws_samecat_before', 'ws_samecat_after'):
                L = final_lock(e)
                inst = L['identity_instance']
                by_ep[i].append(inst)
                info = e['same_category_instances'].get(str(inst)) if inst is not None else None
                if info and info.get('dist_to_target_m') is not None:
                    dists.append(info['dist_to_target_m'])
    multi = {i: v for i, v in by_ep.items() if len(v) >= 2}
    modal_share = [Counter(v).most_common(1)[0][1] / len(v) for v in multi.values()]
    out['repeat_identity'] = {
        'episodes_with_ge2_samecat_wrong_stops': len(multi),
        'all_same_instance': sum(1 for v in multi.values() if len(set(v)) == 1),
        'mean_modal_share': float(np.mean(modal_share)) if modal_share else None,
        'mean_distinct_instances_locked': float(np.mean([len(set(v)) for v in multi.values()])) if multi else None,
    }
    out['wrong_lock_dist_to_target_m'] = {
        'n': len(dists), 'median': float(np.median(dists)) if dists else None,
        'share_le_3m': float(np.mean([d <= 3 for d in dists])) if dists else None,
        'share_ge_8m': float(np.mean([d >= 8 for d in dists])) if dists else None}
    # Order luck: within episodes with both outcomes, does seeing the target before any
    # same-category distractor predict success?
    rows = []
    for r in k1:
        for i, e in data[r]['episodes'].items():
            fv = first_visible_step(e, vis_px)
            fd = first_distractor_visible_step(e, vis_px)
            if fv is None and fd is None:
                order = 'none'
            elif fd is None or (fv is not None and fv < fd):
                order = 'target_first'
            else:
                order = 'distractor_first'
            rows.append((i, order, bool(e['success'])))
    succ_ep = Counter(i for i, _, s in rows if s)
    n_ep = Counter(i for i, _, _ in rows)
    flaky = {i for i in n_ep if 0 < succ_ep[i] < n_ep[i]}
    overall = {o: [s for _, oo, s in rows if oo == o] for o in ('target_first', 'distractor_first', 'none')}
    out['order_overall'] = {o: {'n': len(v), 'sr': float(np.mean(v)) if v else None} for o, v in overall.items()}
    diffs = []
    for i in flaky:
        tf = [s for ii, o, s in rows if ii == i and o == 'target_first']
        df = [s for ii, o, s in rows if ii == i and o == 'distractor_first']
        if tf and df:
            diffs.append(float(np.mean(tf)) - float(np.mean(df)))
    m, lo, hi = paired_bootstrap(diffs, n_boot)
    fl = {o: [s for ii, oo, s in rows if ii in flaky and oo == o] for o in ('target_first', 'distractor_first')}
    orders = defaultdict(set)
    for i, o, _ in rows:
        orders[i].add(o)
    out['order_constant_episodes'] = {'n_episodes': len(orders), 'same_order_in_all_runs': sum(len(v) == 1 for v in orders.values())}
    out['order_flaky'] = {'n_flaky_episodes': len(flaky), 'n_episodes_with_both_orders': len(diffs),
                          'within_episode_sr_diff': m, 'ci': [lo, hi],
                          'sr_target_first': float(np.mean(fl['target_first'])) if fl['target_first'] else None,
                          'sr_distractor_first': float(np.mean(fl['distractor_first'])) if fl['distractor_first'] else None,
                          'n_target_first': len(fl['target_first']), 'n_distractor_first': len(fl['distractor_first'])}
    # Wrong-stop rate by number of same-category instances in the scene (episode-macro).
    by_bin = {}
    for g in K1_GROUPS + ('k3', 'off'):
        runs = group_runs(data, g)
        if not runs:
            continue
        res = {}
        for _, _, name in INST_BINS:
            pairs_ws, pairs_wo, pairs_sr = {}, {}, {}
            for r in runs:
                for i, e in data[r]['episodes'].items():
                    if bin_of(n_scene_instances(e), INST_BINS) != name:
                        continue
                    c = classify(e, vis_px)
                    for pairs, hit in ((pairs_ws, c in ('ws_samecat_before', 'ws_samecat_after')),
                                       (pairs_wo, c == 'ws_other'), (pairs_sr, c == 'success')):
                        n, d = pairs.get(i, (0, 0))
                        pairs[i] = (n + hit, d + 1)
            res[name] = {'samecat_wrong_stop': rate_summary(pairs_ws, n_boot),
                         'other_wrong_stop': rate_summary(pairs_wo, n_boot),
                         'success': rate_summary(pairs_sr, n_boot),
                         'n_episodes': len(pairs_sr)}
        by_bin[g] = res
    out['by_scene_instances'] = by_bin
    # Item 6: wrong stops on objects that are not the goal category.
    rows6 = []
    for g in K1_GROUPS + ('k3',):
        for r in group_runs(data, g):
            for i, e in data[r]['episodes'].items():
                if classify(e, vis_px) != 'ws_other':
                    continue
                L = final_lock(e)
                b = lock_box(e, L)
                top = (b or {}).get('top_instances') or []
                rows6.append({
                    'group': g, 'run': r, 'episode': i, 'category': e['category'], 'path': L.get('path'),
                    'n_scene_instances': n_scene_instances(e),
                    'top_name': top[0]['name'] if top else None,
                    'top_goal_category': top[0].get('goal_category') if top else None,
                    'goal_instance': (b or {}).get('goal_instance'),
                    'goal_frac_of_box': (b or {}).get('goal_instance_frac_of_box'),
                    'target_frac_of_box': (b or {}).get('target_frac_of_box'),
                    'nearest_instance_dist_m': L.get('nearest_instance_dist_m'),
                    'nearest_is_target': L.get('nearest_is_target'),
                    'box_area_frac': (b or {}).get('box_area_frac'),
                    'label_source': (b or {}).get('label_source'),
                    'identity_source': L.get('identity_source')})
    sub = Counter()
    for x in rows6:
        if x['target_frac_of_box'] and x['target_frac_of_box'] >= 0.1:
            sub['target_in_box_not_dominant'] += 1
        elif x['goal_frac_of_box'] and x['goal_frac_of_box'] >= 0.1:
            sub['samecat_in_box_not_dominant'] += 1
        else:
            sub['no_goal_category_in_box'] += 1
    names = Counter((x['category'], x['top_name']) for x in rows6)
    out['noncategory_stops'] = {'n': len(rows6), 'subtypes': dict(sub),
                                'top_names': [[c, n, k] for (c, n), k in names.most_common(40)],
                                'by_category': dict(Counter(x['category'] for x in rows6)),
                                'single_instance_scene': sum(1 for x in rows6 if x['n_scene_instances'] == 1),
                                'rows': rows6}
    return out


def vote_boxes(e, stage):
    return [b for b in e['boxes'] if b.get(stage) in ('yes', 'no', 'unknown') and b.get('frame_check') != 'mismatch']


def analysis_scale(data, n_boot):
    """H3: yes rate by stage x scale x label, episode-macro with cluster bootstrap CIs."""
    out = {}
    combos = [('adm', 'confirmation'), ('def', 'admission'), ('def', 'confirmation'),
              ('k3', 'admission'), ('k3', 'confirmation')]
    for g, stage in combos:
        runs = group_runs(data, g)
        if not runs:
            continue
        res = {}
        for scale_name, key, bins in (('area', 'box_area_frac', AREA_BINS), ('depth', 'top_depth_m', DEPTH_BINS)):
            table = {}
            for lab in LABELS:
                row = {}
                for _, _, bname in bins:
                    pairs = {}
                    for r in runs:
                        for i, e in data[r]['episodes'].items():
                            for b in vote_boxes(e, stage):
                                if b.get('label') != lab or bin_of(b.get(key), bins) != bname:
                                    continue
                                n, d = pairs.get(i, (0, 0))
                                pairs[i] = (n + (b[stage] == 'yes'), d + 1)
                    row[bname] = rate_summary(pairs, n_boot)
                table[lab] = row
            res[scale_name] = table
        # Within-episode contrast for the target: small (<5%) vs large (>=20%) boxes.
        small, large = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        trunc = {True: defaultdict(lambda: [0, 0]), False: defaultdict(lambda: [0, 0])}
        for r in runs:
            for i, e in data[r]['episodes'].items():
                for b in vote_boxes(e, stage):
                    if b.get('label') != 'target' or b.get('box_area_frac') is None:
                        continue
                    y = b[stage] == 'yes'
                    if b['box_area_frac'] < 0.05:
                        small[i][0] += y
                        small[i][1] += 1
                    elif b['box_area_frac'] >= 0.20:
                        large[i][0] += y
                        large[i][1] += 1
                    if b['box_area_frac'] >= 0.05 and b.get('box'):
                        t = trunc[is_truncated(b['box'])][i]
                        t[0] += y
                        t[1] += 1
        both = [i for i in small if i in large]
        diffs = [small[i][0] / small[i][1] - large[i][0] / large[i][1] for i in both]
        m, lo, hi = paired_bootstrap(diffs, n_boot)
        res['target_small_minus_large'] = {'n_episodes': len(both), 'mean_diff': m, 'ci': [lo, hi],
                                           'n_positive': sum(d > 0 for d in diffs), 'n_negative': sum(d < 0 for d in diffs)}
        # Same contrast on depth: target views nearer than 1.5 m vs 1.5-4 m, within episode.
        near, mid = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for r in runs:
            for i, e in data[r]['episodes'].items():
                for b in vote_boxes(e, stage):
                    d = b.get('top_depth_m')
                    if b.get('label') != 'target' or d is None:
                        continue
                    acc = near if d < 1.5 else (mid if d < 4.0 else None)
                    if acc is not None:
                        acc[i][0] += b[stage] == 'yes'
                        acc[i][1] += 1
        both = [i for i in near if i in mid]
        diffs = [near[i][0] / near[i][1] - mid[i][0] / mid[i][1] for i in both]
        m, lo, hi = paired_bootstrap(diffs, n_boot)
        res['target_near_minus_mid_depth'] = {'n_episodes': len(both), 'mean_diff': m, 'ci': [lo, hi],
                                              'n_positive': sum(d > 0 for d in diffs), 'n_negative': sum(d < 0 for d in diffs)}
        # First judgment of each object only: later views are selected by earlier answers
        # (in k=1 runs a yes ends the judging, so only rejected targets get close).
        first = {}
        for lab in LABELS:
            row = {}
            for _, _, bname in DEPTH_BINS:
                pairs = {}
                for r in runs:
                    for i, e in data[r]['episodes'].items():
                        seen = set()
                        for b in sorted(vote_boxes(e, stage), key=lambda b: (b['step'], b.get('box_index') or 0)):
                            key = b.get('gt_instance')
                            if b.get('label') != lab or key is None or key in seen:
                                continue
                            seen.add(key)
                            if bin_of(b.get('top_depth_m'), DEPTH_BINS) != bname:
                                continue
                            n, d = pairs.get(i, (0, 0))
                            pairs[i] = (n + (b[stage] == 'yes'), d + 1)
                row[bname] = rate_summary(pairs, n_boot)
            first[lab] = row
        res['first_view_depth'] = first
        res['target_ge5pct_by_truncation'] = {
            ('truncated' if k else 'not_truncated'): rate_summary({i: tuple(v) for i, v in d.items()}, n_boot)
            for k, d in trunc.items()}
        # Pooled precision of a yes, by area bin.
        prec = {}
        for _, _, bname in AREA_BINS:
            pairs = {}
            for r in runs:
                for i, e in data[r]['episodes'].items():
                    for b in vote_boxes(e, stage):
                        if b[stage] != 'yes' or bin_of(b.get('box_area_frac'), AREA_BINS) != bname:
                            continue
                        n, d = pairs.get(i, (0, 0))
                        pairs[i] = (n + (b.get('label') == 'target'), d + 1)
            lo_, hi_ = cluster_bootstrap(pairs, pooled_rate, n_boot)
            prec[bname] = {'precision': pooled_rate(pairs), 'ci': [lo_, hi_], 'n_yes': sum(d for _, d in pairs.values())}
        res['yes_precision_by_area'] = prec
        out[f'{g}/{stage}'] = res
    return out


def analysis_exposure(data, vis_px, n_boot):
    """H2: what the agent saw and judged before its first lock-vote yes."""
    out = {}
    for g in K1_GROUPS:
        runs = group_runs(data, g)
        # Per-view vote-stage rates by label x area bin (pooled) for the constant-rate model.
        cnt = defaultdict(lambda: [0, 0])
        for r in runs:
            for e in data[r]['episodes'].values():
                for b in vote_boxes(e, 'confirmation'):
                    k = (b.get('label'), bin_of(b.get('box_area_frac'), AREA_BINS))
                    cnt[k][0] += b['confirmation'] == 'yes'
                    cnt[k][1] += 1
        lab_cnt = defaultdict(lambda: [0, 0])
        for (lab, _), (y, n) in cnt.items():
            lab_cnt[lab][0] += y
            lab_cnt[lab][1] += n

        def p_of(b):
            k = (b.get('label'), bin_of(b.get('box_area_frac'), AREA_BINS))
            y, n = cnt.get(k, (0, 0))
            if n >= 20:
                return y / n
            y, n = lab_cnt.get(b.get('label'), (0, 0))
            return y / n if n else 0.0

        rows = []
        for r in runs:
            for i, e in data[r]['episodes'].items():
                vb = sorted(vote_boxes(e, 'confirmation'), key=lambda b: (b['step'], b.get('box_index') or 0))
                first_yes = next((b for b in vb if b['confirmation'] == 'yes'), None)
                upto = first_yes['step'] if first_yes else None
                pre = [b for b in vb if upto is None or b['step'] < upto or b is first_yes]
                first_t = next((k for k, b in enumerate(pre) if b.get('label') == 'target'), None)
                n_d_before_t = sum(1 for b in (pre if first_t is None else pre[:first_t]) if b.get('label') == 'same_category')
                n_o_before_t = sum(1 for b in (pre if first_t is None else pre[:first_t]) if b.get('label') == 'other')
                limit_step = upto if upto is not None else (e['exposure'][-1]['step'] if e['exposure'] else 0)
                seen_inst = set()
                tgt = str(e['target_instance'])
                tgt_frames = dis_frames = 0
                for x in e['exposure']:
                    if x['step'] > limit_step:
                        break
                    vis = {k for k, v in x['same_category_px'].items() if v >= vis_px}
                    seen_inst |= vis - {tgt}
                    tgt_frames += x['target_px'] >= vis_px
                    dis_frames += bool(vis - {tgt})
                model = first_yes_target_prob([(b.get('label') == 'target', p_of(b)) for b in pre]) if first_yes else None
                rows.append({'run': r, 'episode': i, 'first_yes_label': first_yes.get('label') if first_yes else None,
                             'n_samecat_judged_before_target': n_d_before_t, 'n_other_judged_before_target': n_o_before_t,
                             'target_judged_before_first_yes': first_t is not None,
                             'distinct_distractors_seen': len(seen_inst), 'target_frames': tgt_frames,
                             'distractor_frames': dis_frames, 'model_p_target_first': model,
                             'n_scene_instances': n_scene_instances(e)})
        fy = [x for x in rows if x['first_yes_label']]
        lab_counts = Counter(x['first_yes_label'] for x in fy)
        # Observed vs constant-rate model, binned by same-category judgments before the target.
        bins = [(0, 0, '0'), (1, 5, '1-5'), (6, 20, '6-20'), (21, 10 ** 6, '>20')]
        binned = {}
        for lo, hi, name in bins:
            sub = [x for x in fy if lo <= x['n_samecat_judged_before_target'] <= hi]
            pairs = {}
            for x in sub:
                n, d = pairs.get(x['episode'], (0, 0))
                pairs[x['episode']] = (n + (x['first_yes_label'] == 'target'), d + 1)
            s = rate_summary(pairs, n_boot)
            mod = [x['model_p_target_first'] for x in sub if x['model_p_target_first'] is not None]
            binned[name] = {'n_runs': len(sub), 'observed_macro': s['macro'], 'observed_pooled': s['pooled'],
                            'ci': s['ci'], 'model_mean': float(np.mean(mod)) if mod else None}
        before_first = [x for x in rows]
        out[g] = {
            'runs': runs,
            'n_first_yes': len(fy), 'first_yes_labels': dict(lab_counts),
            'observed_target_first': lab_counts.get('target', 0) / len(fy) if fy else None,
            'model_target_first': float(np.mean([x['model_p_target_first'] for x in fy if x['model_p_target_first'] is not None])),
            'per_view_rates': {lab: (y / n if n else None, n) for lab, (y, n) in lab_cnt.items()},
            'binned_by_samecat_before_target': binned,
            'median_distinct_distractors_seen_before_first_yes': float(np.median([x['distinct_distractors_seen'] for x in fy])) if fy else None,
            'share_first_yes_before_target_judged': float(np.mean([not x['target_judged_before_first_yes'] for x in fy])) if fy else None,
            'frames_ratio_distractor_to_target': (sum(x['distractor_frames'] for x in before_first) /
                                                  max(1, sum(x['target_frames'] for x in before_first))),
            'judged_samecat_before_target_median': float(np.median([x['n_samecat_judged_before_target'] for x in fy])) if fy else None,
            'rows': rows,
        }
    return out


def analysis_hard_distractors(data, vis_px, n_boot):
    """F6 on exact labels: yes rates on the instances that caused wrong stops vs on the target."""
    out = {}
    for g in K1_GROUPS:
        runs = group_runs(data, g)
        hard_by_run = defaultdict(lambda: defaultdict(set))   # run -> episode -> instances it wrongly stopped at
        for r in runs:
            for i, e in data[r]['episodes'].items():
                if classify(e, vis_px) in ('ws_samecat_before', 'ws_samecat_after'):
                    hard_by_run[r][i].add(final_lock(e)['identity_instance'])
        p_hard, p_tgt, p_other_sc = {}, {}, {}
        for r in runs:
            # Leave-one-run-out: an instance is "hard" in run r if it caused a wrong stop in
            # another run of the configuration, so its yes rate in r is not selected on r's yes.
            hard = defaultdict(set)
            for r2 in runs:
                if r2 != r:
                    for i, s in hard_by_run[r2].items():
                        hard[i] |= s
            for i, e in data[r]['episodes'].items():
                for b in vote_boxes(e, 'confirmation'):
                    lab = b.get('label')
                    y = b['confirmation'] == 'yes'
                    if lab == 'target':
                        tgt = p_tgt
                    elif lab == 'same_category' and b.get('gt_instance') in hard.get(i, ()):
                        tgt = p_hard
                    elif lab == 'same_category':
                        tgt = p_other_sc
                    else:
                        continue
                    n, d = tgt.get(i, (0, 0))
                    tgt[i] = (n + y, d + 1)
        both = [i for i in p_hard if i in p_tgt and p_hard[i][1] and p_tgt[i][1]]
        out[g] = {'hard_distractor': rate_summary(p_hard, n_boot), 'target': rate_summary(p_tgt, n_boot),
                  'other_same_category': rate_summary(p_other_sc, n_boot),
                  'episodes_with_both': len(both),
                  'target_rate_above_hard': sum(p_tgt[i][0] / p_tgt[i][1] > p_hard[i][0] / p_hard[i][1] for i in both),
                  'target_rate_equal_hard': sum(p_tgt[i][0] / p_tgt[i][1] == p_hard[i][0] / p_hard[i][1] for i in both)}
    return out


def analysis_correlation(data, n_boot, n_perm, seed=0):
    """H4: are repeated yes answers on one distractor instance independent?"""
    out = {}
    for g, stage in (('adm', 'confirmation'), ('def', 'admission'), ('def', 'confirmation'), ('k3', 'confirmation')):
        runs = group_runs(data, g)
        if not runs:
            continue
        seqs, per_run_inst = [], defaultdict(dict)
        for r in runs:
            for i, e in data[r]['episodes'].items():
                inst = defaultdict(list)
                for b in sorted(vote_boxes(e, stage), key=lambda b: (b['step'], b.get('box_index') or 0)):
                    if b.get('label') == 'same_category' and b.get('gt_instance') is not None:
                        inst[b['gt_instance']].append(int(b[stage] == 'yes'))
                for k, s in inst.items():
                    seqs.append(s)
                    per_run_inst[r][(i, k)] = s
        all_y = sum(sum(s) for s in seqs)
        all_n = sum(len(s) for s in seqs)
        p = all_y / all_n if all_n else 0.0
        tails = {}
        for kk in (1, 2, 3):
            elig = [s for s in seqs if len(s) >= kk]
            obs = sum(sum(s) >= kk for s in elig)
            exp = sum(binom_tail(len(s), p, kk) for s in elig)
            tails[f'>={kk}'] = {'n_instances': len(elig), 'observed': obs, 'expected_independent': exp}
        # Intra-class correlation (ANOVA estimator for binary outcomes) over instances with >= 2 views.
        s2 = [s for s in seqs if len(s) >= 2]
        icc = None
        if len(s2) > 1:
            N = sum(len(s) for s in s2)
            k = len(s2)
            pbar = sum(sum(s) for s in s2) / N
            msb = sum(len(s) * (np.mean(s) - pbar) ** 2 for s in s2) / (k - 1)
            msw = sum(sum((v - np.mean(s)) ** 2 for v in s) for s in s2) / (N - k)
            n0 = (N - sum(len(s) ** 2 for s in s2) / N) / (k - 1)
            icc = float((msb - msw) / (msb + (n0 - 1) * msw)) if (msb + (n0 - 1) * msw) > 0 else None
        lag = lag1_stats(seqs)
        # Cross-run consistency of per-instance yes rates (same episode, same instance).
        rhos = []
        rl = list(per_run_inst)
        for a in range(len(rl)):
            for b_ in range(a + 1, len(rl)):
                xs, ys = [], []
                for key, s in per_run_inst[rl[a]].items():
                    t = per_run_inst[rl[b_]].get(key)
                    if t and len(s) >= 3 and len(t) >= 3:
                        xs.append(np.mean(s))
                        ys.append(np.mean(t))
                if len(xs) >= 10:
                    from scipy.stats import spearmanr
                    rho = spearmanr(xs, ys).correlation
                    rhos.append({'runs': [rl[a], rl[b_]], 'n_instances': len(xs), 'spearman': float(rho)})
        # Does an instance that got a yes in one run get a yes in another run of the same config?
        cond = {'any_yes_given_yes_elsewhere': [0, 0], 'any_yes_given_no_yes_elsewhere': [0, 0]}
        for a in rl:
            for b_ in rl:
                if a == b_:
                    continue
                for key, s in per_run_inst[a].items():
                    t = per_run_inst[b_].get(key)
                    if not t:
                        continue
                    k = 'any_yes_given_yes_elsewhere' if sum(s) else 'any_yes_given_no_yes_elsewhere'
                    cond[k][0] += bool(sum(t))
                    cond[k][1] += 1
        out[f'{g}/{stage}'] = {'per_view_yes': p, 'n_instances': len(seqs), 'n_views': all_n, 'tails': tails,
                               'icc': icc, 'lag1': {'p_yes_after_yes': lag[0], 'p_yes_after_no': lag[1],
                                                    'n_after_yes': lag[2], 'n_after_no': lag[3]},
                               'lag1_shuffle': lag1_shuffle_test(seqs, n_perm, seed),
                               'cross_run_spearman': rhos,
                               'cross_run_any_yes': {k: {'rate': (v[0] / v[1] if v[1] else None), 'n': v[1]}
                                                     for k, v in cond.items()}}
    return out


def analysis_deferral(data, vis_px, n_boot):
    """Item 7: if the agent had not stopped at its wrong lock, would it have seen the target?"""
    out = {}
    all_runs = sorted(data)
    T_MAX = max((x['step'] for r in all_runs for e in data[r]['episodes'].values() for x in e['exposure']), default=982)
    # Subjects: every run-episode; event = target first visible; censored at the first lock
    # (afterwards the agent approaches the locked object instead of exploring) or at the end.
    subjects = defaultdict(list)
    for r in all_runs:
        for i, e in data[r]['episodes'].items():
            fv = first_visible_step(e, vis_px)
            end = e['exposure'][-1]['step'] if e['exposure'] else 0
            cens = e['locks'][0]['step'] if e['locks'] else end
            if fv is not None and fv <= cens:
                subjects[i].append((fv, True))
            else:
                subjects[i].append((cens, False))
    curves = {i: kaplan_meier(v) for i, v in subjects.items()}
    last_obs = {i: max(t for t, _ in v) for i, v in subjects.items()}
    k1 = [r for g in K1_GROUPS for r in group_runs(data, g)]
    rows = []
    for r in k1:
        for i, e in data[r]['episodes'].items():
            if classify(e, vis_px) != 'ws_samecat_before':
                continue
            tl = final_lock(e)['step']
            c = curves[i]
            s_tl = km_survival_at(c, tl)
            # Lower bound: no further sightings after the last observed step of this episode.
            # Upper bound: if no run observed the episode to the end, the unobserved tail could
            # contain a sighting for every remaining subject.
            p_lb = 1 - km_survival_at(c, T_MAX) / s_tl if s_tl > 0 else 0.0
            p_ub = p_lb if last_obs[i] >= T_MAX - 5 else 1.0
            rows.append({'run': r, 'episode': i, 'lock_step': tl, 'p_seen_later_lb': p_lb, 'p_seen_later_ub': p_ub,
                         'last_observed_step': last_obs[i]})
    n_runs = len(k1)
    out['wrong_stop_before_seen_per_run'] = len(rows) / n_runs if n_runs else None
    out['expected_seen_later_per_run'] = {'lower': sum(x['p_seen_later_lb'] for x in rows) / n_runs,
                                          'upper': sum(x['p_seen_later_ub'] for x in rows) / n_runs}
    out['share_with_tail_observed_to_end'] = float(np.mean([x['last_observed_step'] >= T_MAX - 5 for x in rows])) if rows else None
    out['t_max'] = T_MAX
    out['median_lock_step'] = float(np.median([x['lock_step'] for x in rows])) if rows else None
    # Conversion: once the target is visible before any lock, how often does a k=1 run succeed?
    conv = {}
    for g in K1_GROUPS:
        pairs = {}
        for r in group_runs(data, g):
            for i, e in data[r]['episodes'].items():
                fv = first_visible_step(e, vis_px)
                first_lock = e['locks'][0]['step'] if e['locks'] else None
                if fv is not None and (first_lock is None or fv <= first_lock):
                    n, d = pairs.get(i, (0, 0))
                    pairs[i] = (n + bool(e['success']), d + 1)
        conv[g] = rate_summary(pairs, n_boot)
    out['p_success_given_target_seen_before_lock'] = conv
    # Ground check: what the k=3 run did on episodes where k=1 runs stopped early on a distractor.
    k3 = group_runs(data, 'k3')
    if k3:
        eps = sorted({x['episode'] for x in rows})
        out['k3_on_wrong_stop_episodes'] = {'n_episodes': len(eps),
                                            'classes': dict(Counter(classify(data[k3[0]]['episodes'][i], vis_px) for i in eps))}
    # Pooled KM curves for the figure: episodes with a wrong stop before seen vs the rest.
    ws_eps = {x['episode'] for x in rows}
    pooled = {'wrong_stop_episodes': kaplan_meier([s for i in ws_eps for s in subjects[i]]),
              'other_episodes': kaplan_meier([s for i in subjects if i not in ws_eps for s in subjects[i]])}
    out['km_pooled'] = {k: [[t, s] for t, s in v] for k, v in pooled.items()}
    out['km_at_risk'] = {k: {t: sum(1 for i in (ws_eps if k == 'wrong_stop_episodes' else set(subjects) - ws_eps)
                                    for tt, _ in subjects[i] if tt >= t) for t in (100, 300, 600, 900)}
                         for k in pooled}
    out['km_cum_seen'] = {k: {t: 1 - km_survival_at(v, t) for t in (112, 300, 600, T_MAX)} for k, v in pooled.items()}
    out['rows'] = rows
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
LABEL_COLOR = {'target': SERIES[0], 'same_category': SERIES[1], 'other': SERIES[2]}
LABEL_ZH = {'target': '目标', 'same_category': '同类其他实例', 'other': '非同类'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'SimHei'):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            break
    plt.rcParams.update({'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'axes.unicode_minus': False, 'font.size': 10})
    return plt


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_scale(scale, path):
    """Rows: stage; columns: area / depth. Solid = episode-macro with 95% CI, dashed = pooled."""
    plt = _plt()
    rows = [(k, t) for k, t in (('adm/confirmation', '准入关闭（基线）：4 m 内锁定核验'),
                                ('def/admission', '默认：整帧候选准入')) if k in scale]
    cols = [('area', AREA_BINS, '候选框占画面比例'), ('depth', DEPTH_BINS, '框内主体的真实深度')]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.2, 3.3 * len(rows)), sharey=True, squeeze=False)
    for ri, (key, title) in enumerate(rows):
        for ci, (scale_name, bins, xlabel) in enumerate(cols):
            ax = axes[ri][ci]
            names = [b[2] for b in bins]
            for j, lab in enumerate(LABELS):
                row = scale[key][scale_name][lab]
                xs, ys, lo, hi, xp, yp = [], [], [], [], [], []
                for k, n in enumerate(names):
                    s_ = row[n]
                    if s_['macro'] is None or s_['n_episodes'] < 5:
                        continue
                    x = k + (j - 1) * 0.09
                    xs.append(x)
                    ys.append(s_['macro'])
                    lo.append(s_['macro'] - (s_['ci'][0] if s_['ci'][0] is not None else s_['macro']))
                    hi.append((s_['ci'][1] if s_['ci'][1] is not None else s_['macro']) - s_['macro'])
                    xp.append(x)
                    yp.append(s_['pooled'])
                ax.errorbar(xs, ys, yerr=[lo, hi], color=LABEL_COLOR[lab], marker='o', markersize=5, linewidth=2,
                            capsize=0, elinewidth=1, label=LABEL_ZH[lab] + '（按任务平均）')
                if lab != 'other':
                    ax.plot(xp, yp, color=LABEL_COLOR[lab], linestyle='--', linewidth=1.2, marker='s', markersize=3,
                            label=LABEL_ZH[lab] + '（混合计算）')
            ax.set_xticks(range(len(names)), names)
            ax.set_xlabel(xlabel)
            ax.set_ylim(0, 0.8)
            if ci == 0:
                ax.set_ylabel('判 yes 的比例')
            ax.set_title(title, fontsize=10, loc='left')
            _style(ax)
    axes[0][1].legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_taxonomy(tax, path):
    plt = _plt()
    groups = [g for g in GROUP_ORDER if g in tax]
    merged = [('成功', ['success']),
              ('目标可见前错停（同类）', ['ws_samecat_before']),
              ('其他错停', ['ws_samecat_after', 'ws_other', 'stop_no_lock']),
              ('锁对目标但失败', ['stop_far_target_lock', 'lim_target_locked']),
              ('目标从未可见', ['lim_never_visible']),
              ('可见但未检出/未接受', ['lim_not_boxed', 'lim_not_accepted', 'lim_accepted_no_lock', 'lim_wrong_lock'])]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for gi, g in enumerate(groups):
        left = 0.0
        for ci, (name, keys) in enumerate(merged):
            v = sum(tax[g]['mean'][k] for k in keys)
            ax.barh(gi, v, left=left, color=SERIES[ci], edgecolor=SURFACE, linewidth=2, height=0.6,
                    label=name if gi == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, gi, f'{v:.0f}', ha='center', va='center', fontsize=8, color=INK)
            left += v
    ax.set_yticks(range(len(groups)), [GROUP_NAME[g] + f'（{len(tax[g]["runs"])} 次）' for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('每次运行的任务数（100 条，按运行平均）')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.25))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_exposure(expo, wl, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    ax = axes[0]
    for j, g in enumerate(K1_GROUPS):
        b = expo[g]['binned_by_samecat_before_target']
        all_names = list(b)
        keep = [k for k, n in enumerate(all_names) if b[n]['observed_macro'] is not None and b[n]['n_runs'] >= 5]
        names = [all_names[k] for k in keep]
        xs = [k + (j - 0.5) * 0.1 for k in keep]
        ys = [b[n]['observed_macro'] for n in names]
        lo = [b[n]['observed_macro'] - b[n]['ci'][0] if b[n]['ci'][0] is not None else 0 for n in names]
        hi = [b[n]['ci'][1] - b[n]['observed_macro'] if b[n]['ci'][1] is not None else 0 for n in names]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[j], marker='o', markersize=5, linewidth=2, elinewidth=1,
                    label=GROUP_NAME[g] + ' 实测')
        ax.plot(xs, [b[n]['model_mean'] for n in names], color=SERIES[j], linestyle='--', linewidth=1.5,
                label=GROUP_NAME[g] + ' 恒定判定率模型')
        ax.set_xticks(range(len(all_names)), all_names)
    ax.set_xlabel('第一次判到目标之前，已判过的同类干扰物框数')
    ax.set_ylabel('第一张锁定票投给目标的比例')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    _style(ax)
    ax = axes[1]
    names = [b[2] for b in INST_BINS]
    for j, g in enumerate(K1_GROUPS):
        res = wl['by_scene_instances'][g]
        xs = [k + (j - 0.5) * 0.1 for k in range(len(names))]
        ys = [res[n]['samecat_wrong_stop']['macro'] or 0 for n in names]
        lo = [ys[k] - (res[n]['samecat_wrong_stop']['ci'][0] or ys[k]) for k, n in enumerate(names)]
        hi = [(res[n]['samecat_wrong_stop']['ci'][1] or ys[k]) - ys[k] for k, n in enumerate(names)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[j], marker='o', markersize=5, linewidth=2, elinewidth=1,
                    label=GROUP_NAME[g])
    ax.set_xticks(range(len(names)), [f'{n}\n({wl["by_scene_instances"]["adm"][n]["n_episodes"]} 条)' for n in names])
    ax.set_xlabel('场景中同类实例数')
    ax.set_ylabel('停在同类干扰物上的比例')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_deferral(defr, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for j, (k, name) in enumerate((('wrong_stop_episodes', '曾在目标可见前错停的任务'), ('other_episodes', '其余任务'))):
        c = defr['km_pooled'][k]
        xs, ys = [0], [0.0]
        for t, s in c:
            xs += [t, t]
            ys += [ys[-1], 1 - s]
        ax.plot(xs, ys, color=SERIES[j], linewidth=2, label=name)
    ax.set_xlabel('策略步')
    ax.set_ylabel('目标已清晰可见的累计概率（KM）')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1000)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _pct(x, nd=1):
    return '—' if x is None else f'{100 * x:.{nd}f}%'


def _ci(ci, nd=1):
    if not ci or ci[0] is None:
        return ''
    return f' [{100 * ci[0]:.{nd}f}, {100 * ci[1]:.{nd}f}]'


def write_tables(res, path):
    L = ['# S3 自动生成表格', '', f"可见阈值：{res['args']['vis_px']} 像素。区间为按任务的 95% bootstrap 区间。", '']
    tax = res['taxonomy']
    L += ['## 失败分类（每次运行的平均任务数 ± 标准差）', '', '| 类别 | ' + ' | '.join(GROUP_NAME[g] for g in GROUP_ORDER if g in tax) + ' |',
          '|---|' + '---|' * sum(g in tax for g in GROUP_ORDER)]
    for k, name in TAXONOMY:
        cells = []
        for g in GROUP_ORDER:
            if g in tax:
                cells.append(f"{tax[g]['mean'][k]:.1f} ± {tax[g]['sd'][k]:.1f}")
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    ns = tax['never_solved_k1']
    tot = sum(ns['failure_runs'].values())
    L += ['', f"六次 k=1 运行中从未成功的任务：{len(ns['episodes'])} 条，共 {tot} 次失败。", '',
          '| 类别 | 失败次数 | 占比 | 作为该任务最常见类别的任务数 |', '|---|---|---|---|']
    for k, name in TAXONOMY:
        if ns['failure_runs'].get(k):
            L.append(f"| {name} | {ns['failure_runs'][k]} | {_pct(ns['failure_runs'][k] / tot)} | {ns['modal_class'].get(k, 0)} |")
    sc = res['scale']
    for key in sc:
        L += ['', f'## 尺度：{key}（判 yes 比例，按任务平均 [95% 区间]；括号内为混合计算 / 框数 / 任务数）', '']
        for scale_name, bins in (('area', AREA_BINS), ('depth', DEPTH_BINS)):
            L += [f'| {scale_name} | ' + ' | '.join(b[2] for b in bins) + ' |', '|---|' + '---|' * len(bins)]
            for lab in LABELS:
                cells = []
                for b in bins:
                    s = sc[key][scale_name][lab][b[2]]
                    cells.append(f"{_pct(s['macro'])}{_ci(s['ci'])} ({_pct(s['pooled'])} / {s['n_boxes']} / {s['n_episodes']})")
                L.append(f'| {LABEL_ZH[lab]} | ' + ' | '.join(cells) + ' |')
            L.append('')
        t = sc[key]['target_small_minus_large']
        L.append(f"目标框 <5% 减去 ≥20% 的任务内差值：{_pct(t['mean_diff'])}{_ci(t['ci'])}，{t['n_episodes']} 条任务"
                 f"（正 {t['n_positive']}，负 {t['n_negative']}）。")
        t = sc[key]['target_near_minus_mid_depth']
        L.append(f"目标框深度 <1.5 m 减去 1.5–4 m 的任务内差值：{_pct(t['mean_diff'])}{_ci(t['ci'])}，{t['n_episodes']} 条任务"
                 f"（正 {t['n_positive']}，负 {t['n_negative']}）。")
        L += ['', '每个物体只取第一次判定（按深度）：', '',
              '| 标签 | ' + ' | '.join(b[2] for b in DEPTH_BINS) + ' |', '|---|' + '---|' * len(DEPTH_BINS)]
        for lab in LABELS:
            cells = []
            for b in DEPTH_BINS:
                s = sc[key]['first_view_depth'][lab][b[2]]
                cells.append(f"{_pct(s['macro'])}{_ci(s['ci'])} ({s['n_boxes']} / {s['n_episodes']})")
            L.append(f'| {LABEL_ZH[lab]} | ' + ' | '.join(cells) + ' |')
        L.append('')
        tr = sc[key]['target_ge5pct_by_truncation']
        L.append(f"目标框 ≥5% 时：贴边 {_pct(tr['truncated']['macro'])}（{tr['truncated']['n_boxes']} 框），"
                 f"不贴边 {_pct(tr['not_truncated']['macro'])}（{tr['not_truncated']['n_boxes']} 框）。")
        pr = sc[key]['yes_precision_by_area']
        L.append('yes 的精确率（混合计算）：' + '，'.join(f"{n} {_pct(v['precision'])}{_ci(v['ci'])}（{v['n_yes']} 个 yes）"
                                                    for n, v in pr.items()) + '。')
    Path(path).write_text('\n'.join(L) + '\n')


# --------------------------------------------------------------------------------------
def _strip(obj):
    """JSON-safe copy without the bulky per-row lists."""
    if isinstance(obj, dict):
        return {str(k): _strip(v) for k, v in obj.items() if k != 'rows'}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/verification_set')
    ap.add_argument('--results-root', default='data')
    ap.add_argument('--out', default='outputs/analysis_verification_set')
    ap.add_argument('--runs', nargs='*', help='Subset of run timestamps (default: all in the manifest)')
    ap.add_argument('--vis-px', type=int, default=2000)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_set(REPO / args.root if not Path(args.root).is_absolute() else args.root,
                    REPO / args.results_root if not Path(args.results_root).is_absolute() else args.results_root,
                    set(args.runs) if args.runs else None)
    res = {'args': vars(args), 'runs': {r: data[r]['group'] for r in sorted(data)}}
    res['taxonomy'] = analysis_taxonomy(data, args.vis_px)
    res['taxonomy_sensitivity'] = {px: {g: v['mean'] for g, v in analysis_taxonomy(data, px).items() if g in GROUP_ORDER}
                                   for px in (500, 5000)}
    res['wrong_locks'] = analysis_wrong_locks(data, args.vis_px, args.boot)
    res['scale'] = analysis_scale(data, args.boot)
    res['exposure'] = analysis_exposure(data, args.vis_px, args.boot)
    res['hard_distractors'] = analysis_hard_distractors(data, args.vis_px, args.boot)
    res['correlation'] = analysis_correlation(data, args.boot, args.perm)
    res['deferral'] = analysis_deferral(data, args.vis_px, args.boot)
    (out / 'results.json').write_text(json.dumps(_strip(res), ensure_ascii=False, indent=1))
    with open(out / 'rows_noncategory_stops.jsonl', 'w') as fh:
        for x in res['wrong_locks']['noncategory_stops']['rows']:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    write_tables(res, out / 'tables.md')
    if not args.no_figures:
        fig_scale(res['scale'], out / 'fig_scale.png')
        fig_taxonomy(res['taxonomy'], out / 'fig_taxonomy.png')
        fig_exposure(res['exposure'], res['wrong_locks'], out / 'fig_exposure.png')
        fig_deferral(res['deferral'], out / 'fig_deferral.png')
    print(f'wrote {out}')
    return res


if __name__ == '__main__':
    main()
