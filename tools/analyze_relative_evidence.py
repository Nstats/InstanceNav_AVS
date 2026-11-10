#!/usr/bin/env python
"""Does relative evidence let a stop rule stop earlier than "explore the full budget, then pick
the best", without losing success? (实验计划.md direction C, corrected S7 pre-study, 2026-11-04)

Corrections relative to analyze_m1m2_candidates.py / replay_rules.py (see the 11-04 review):
  * the pool now contains EVERY detection, including those the model rejected ("no" boxes are
    never kept in the collect run, so the old kept-only pool had no rejected candidate at all);
  * every rule is averaged over all 100 episodes (no silent exclusion of empty episodes);
  * a stop is only allowed on a candidate the agent can still reach with the remaining budget,
    and its payoff depends on how far the stopped candidate is from the goal instance.

Pool: greedy 1 m clustering of all boxes by est_gps in step order (as in F13/F14).
Candidate state: yes if any yes, no if judged and never yes, else unverified.

Payoff of stopping on candidate c (calibrated on the first locks of the six k=1 runs,
P(success | lock-point distance to the goal), which includes later stuck losses):
  [0,1) 0.89  [1,1.5) 0.83  [1.5,3) 0.42  [3,5) 0.04  [5,inf) 0.06
Reachability: walking to a candidate d metres away costs about 7.2 d policy steps on the six
runs (lock-to-stop); a stop needs remaining >= 8 d + 10.

M1 (calibrated candidate probability q_i): a logistic model of "candidate is the goal" on
agent-side candidate features, fitted on the training scenes of a scene-grouped 5-fold split.
M2 (unexplored mass): U in {0, max(0, c - n_seen), frontier unknown/known, both} x budget discount.
M3: P1 = q_lead / (sum_i q_i + alpha * U); stop on the leader when it is a yes candidate,
reachable, and P1 >= tau. Fallback at T_fb: stop on the best reachable yes candidate.

Rules compared (all with the same fallback family and out-of-fold parameter choice):
  first_match    stop on the first candidate that gets a yes
  full_budget    no early stop, fallback at T_fb (T_fb chosen on training scenes)
  count_k        stop on the best yes candidate once >= k candidates have been rejected (F13)
  p1_<U>         the unified rule with the given U

Run: PYTHONPATH=habitat-lab:. $PSGNAV_PYTHON tools/analyze_relative_evidence.py
"""
import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.analyze_verification_set import load_set  # noqa: E402
from tools.build_verification_set import world_to_gps  # noqa: E402

RUN = '20261103_011329_142191'
RADIUS = 1.0
SIGHT_GAP = 10
PAY_BINS = ((1.0, 0.89), (1.5, 0.83), (3.0, 0.42), (5.0, 0.04), (1e9, 0.06))
WALK_A, WALK_B = 8.0, 10.0
FB_GRID = (300, 500, 700, 800, 860, 900, 930)
RNG = np.random.default_rng(20261104)


def payoff_of_distance(d):
    for hi, p in PAY_BINS:
        if d < hi:
            return p
    return PAY_BINS[-1][1]


# --------------------------------------------------------------------------------------
# Per-episode moment table (agent side), with GT only in 'tdist' (evaluation / calibration)
# --------------------------------------------------------------------------------------
class Pool:
    def __init__(self):
        self.c = []   # dict(sum_gps, n, yes, no, steps, dmin, last_yes_gps)

    def add(self, b):
        g = b.get('est_gps')
        if not g:
            return
        for c in self.c:
            if math.dist(c['seed'], g) <= RADIUS:
                break
        else:
            c = {'seed': list(g), 'sx': 0.0, 'sy': 0.0, 'n': 0, 'yes': 0, 'no': 0, 'steps': [],
                 'dmin': 99.0, 'yes_gps': None}
            self.c.append(c)
        c['sx'] += g[0]
        c['sy'] += g[1]
        c['n'] += 1
        a = b.get('confirmation') or b.get('admission')
        if a == 'yes':
            c['yes'] += 1
            c['yes_gps'] = list(g)
        elif a == 'no':
            c['no'] += 1
        c['steps'].append(b['step'])
        # Agent-side distance only (episode_moments sets agent_dist from the pose and est_gps).
        # Never top_depth_m: it is the depth of the GT-labelled pixels in the box.
        d = b.get('agent_dist') or b.get('est_distance_m')
        if d:
            c['dmin'] = min(c['dmin'], float(d))


def sightings(steps):
    n, last = 0, None
    for s in sorted(steps):
        if last is None or s - last >= SIGHT_GAP:
            n += 1
        last = s
    return n


def cand_pos(c):
    return c['yes_gps'] or [c['sx'] / c['n'], c['sy'] / c['n']]


def cand_feats(c):
    st_yes = 1.0 if c['yes'] else 0.0
    st_no = 1.0 if (not c['yes'] and c['no']) else 0.0
    return [st_yes, st_no, math.log1p(c['yes']), math.log1p(c['no']),
            math.log(max(0.5, c['dmin'])), math.log1p(sightings(c['steps']))]


def episode_moments(ep, ds_ep):
    tgt = (ep['same_category_instances'] or {}).get(str(ep['target_instance']))
    tg = tgt['gps'] if tgt else None
    sp, sr = ds_ep['start_position'], ds_ep['start_rotation']
    pos = {}
    for f in ep['frames']:
        pos[f['step']] = world_to_gps(f['position'], sp, sr)
    end = max(pos) if pos else 982
    snaps = sorted(ep.get('collect_snapshots') or [], key=lambda s: s['step'])
    by_step = defaultdict(list)
    for b in ep['boxes']:
        if b.get('step') is not None and b.get('est_gps'):
            if b['step'] in pos:
                b['agent_dist'] = float(np.linalg.norm(np.asarray(pos[b['step']]) - np.asarray(b['est_gps'])))
            by_step[b['step']].append(b)
    times = sorted(set(by_step) | set(range(10, end + 1, 10)))
    pool, out, si, snap = Pool(), [], 0, None
    last_pos = None
    for t in times:
        for b in by_step.get(t, []):
            pool.add(b)
        while si < len(snaps) and snaps[si]['step'] <= t:
            snap = snaps[si]
            si += 1
        if t in pos:
            last_pos = pos[t]
        if not pool.c or last_pos is None:
            continue
        cov = (snap or {}).get('coverage') or {}
        fr = (snap or {}).get('frontiers') or {}
        known, gain = cov.get('known_m2'), fr.get('gain_union_m2')
        X = np.array([cand_feats(c) for c in pool.c])
        P = [cand_pos(c) for c in pool.c]
        out.append({
            't': t, 'remaining': end - t, 'X': X,
            'yes': X[:, 0] > 0, 'no': X[:, 1] > 0,
            'dagent': np.array([math.dist(last_pos, p) for p in P]),
            'tdist': np.array([math.dist(tg, p) if tg else 99.0 for p in P]),
            'n_seen': len(pool.c), 'n_rej': int((X[:, 1] > 0).sum()),
            'gain_ratio': (gain / known) if (gain is not None and known) else 0.0,
        })
    return out, end


# --------------------------------------------------------------------------------------
# Calibration (M1) and rules
# --------------------------------------------------------------------------------------
def fit_logit(X, y, l2=1.0, iters=500):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 / len(y) * np.r_[w[:-1], 0.0]
        w -= 1.0 * g
    return mu, sd, w


def predict(model, X):
    mu, sd, w = model
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Z @ w))


def u_value(m, kind, c_const, end):
    disc = max(0.0, m['remaining']) / max(1.0, end)
    base = max(0.0, c_const - m['n_seen'])
    if kind == 'none':
        return 0.0
    if kind == 'count':
        return base * disc
    if kind == 'frontier':
        return m['gain_ratio'] * disc
    if kind == 'both':
        return (base + m['gain_ratio']) * disc
    raise ValueError(kind)


def reachable(m):
    return m['remaining'] >= WALK_A * m['dagent'] + WALK_B


def best_yes(m, q):
    ok = m['yes'] & reachable(m)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    return int(idx[np.argmax(q[idx])])


def run_episode(moments, end, rule, p, q_list):
    """Return (payoff, stop_step, steps_used, on_target_within_1m)."""
    for m, q in zip(moments, q_list):
        if m['t'] >= p['T_fb']:
            break
        if rule == 'first_match':
            if m['yes'].any():
                ok = m['yes'] & reachable(m)
                if ok.any():
                    i = int(np.flatnonzero(ok)[np.argmax(q[ok])])
                    return _stop(m, i)
            continue
        if rule == 'full_budget':
            continue
        if rule == 'count':
            if m['n_rej'] >= p['k']:
                i = best_yes(m, q)
                if i is not None:
                    return _stop(m, i)
            continue
        # unified P1 rule
        lead = int(np.argmax(q))
        if not (m['yes'][lead] and reachable(m)[lead]):
            continue
        p1 = q[lead] / (q.sum() + p['alpha'] * u_value(m, p['U'], p['c'], end))
        if p1 >= p['tau']:
            return _stop(m, lead)
    # fallback: the last moment before T_fb with a reachable yes candidate
    cand = None
    for m, q in zip(moments, q_list):
        if m['t'] >= p['T_fb']:
            break
        i = best_yes(m, q)
        if i is not None:
            cand = (m, i)
    if cand is None:
        return 0.0, None, end, False
    return _stop(*cand)


def _stop(m, i):
    d = float(m['tdist'][i])
    walk = WALK_A * float(m['dagent'][i])
    return payoff_of_distance(d), m['t'], m['t'] + walk, d < 1.0


def grid(rule):
    fbs = [{'T_fb': t} for t in FB_GRID]
    if rule in ('first_match', 'full_budget'):
        return fbs
    if rule == 'count':
        return [{**f, 'k': k} for f in fbs for k in (1, 2, 3, 4, 6, 8, 12)]
    U = rule.split('_', 1)[1]
    out = []
    for f in fbs:
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            if U == 'none':
                out.append({**f, 'U': 'none', 'alpha': 0.0, 'c': 0.0, 'tau': tau})
                continue
            for alpha in (0.05, 0.1, 0.2, 0.4, 0.8):
                for c in ((6.0, 10.0, 15.0) if U in ('count', 'both') else (0.0,)):
                    out.append({**f, 'U': U, 'alpha': alpha, 'c': c, 'tau': tau})
    return out


RULES = ('first_match', 'full_budget', 'count', 'p1_none', 'p1_count', 'p1_frontier', 'p1_both')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--out', default=str(REPO / 'outputs' / 'analysis_relative_evidence_20261104'))
    args = ap.parse_args()

    data = load_set(REPO / 'data' / 'verification_set', REPO / 'data', runs=[args.run])
    eps = data[args.run]['episodes']
    ents = json.load(gzip.open(REPO / 'data/datasets/instancenav/val10/val10.json.gz'))['episodes']
    ds = {(e['scene_id'].split('/')[-1], str(e['episode_id']), str(e['goal_object_id'])): e for e in ents}
    M, END, SCENE = {}, {}, {}
    for i, e in sorted(eps.items()):
        M[i], END[i] = episode_moments(e, ds[tuple(e['key'])])
        SCENE[i] = e['key'][0]
    tasks = sorted(eps)
    print(f'[moments] {sum(len(v) for v in M.values())} over {len(tasks)} episodes', flush=True)

    # Ceilings on this pool (GT, evaluation only).
    ever_yes_target = sum(any((m['yes'] & (m['tdist'] < 1.0)).any() for m in M[i]) for i in tasks)

    # Scene-grouped folds.
    scenes = sorted(set(SCENE.values()))
    RNG.shuffle(scenes)
    fold_of = {s: k % 5 for k, s in enumerate(scenes)}

    oof = {r: {} for r in RULES}
    chosen = {r: [] for r in RULES}
    calib_rows = []
    for f in range(5):
        train = [i for i in tasks if fold_of[SCENE[i]] != f]
        test = [i for i in tasks if fold_of[SCENE[i]] == f]
        # M1 calibration on training scenes: candidates sampled every 50 steps.
        Xs, ys = [], []
        for i in train:
            for m in M[i]:
                if m['t'] % 50 == 0:
                    Xs.append(m['X'])
                    ys.append((m['tdist'] < 1.0).astype(float))
        model = fit_logit(np.vstack(Xs), np.concatenate(ys))
        Q = {i: [predict(model, m['X']) for m in M[i]] for i in tasks}
        for i in test:
            for m, q in zip(M[i], Q[i]):
                if m['t'] % 50 == 0:
                    calib_rows += list(zip(q, (m['tdist'] < 1.0), m['X'][:, 0], m['X'][:, 1]))
        for rule in RULES:
            base = 'p1' if rule.startswith('p1') else rule
            best, best_v = None, -1
            for p in grid(rule):
                v = np.mean([run_episode(M[i], END[i], base, p, Q[i])[0] for i in train])
                if v > best_v:
                    best, best_v = p, v
            chosen[rule].append(best)
            for i in test:
                oof[rule][i] = run_episode(M[i], END[i], base, best, Q[i])
        print(f'[fold {f}] done', flush=True)

    # Summaries with paired bootstrap against full_budget and first_match.
    res = {'n_episodes': len(tasks), 'ever_yes_on_target': ever_yes_target, 'rules': {}}
    idx = RNG.integers(0, len(tasks), size=(4000, len(tasks)))
    for rule in RULES:
        pay = np.array([oof[rule][i][0] for i in tasks])
        used = np.array([oof[rule][i][2] for i in tasks])
        stops = [oof[rule][i][1] for i in tasks if oof[rule][i][1] is not None]
        on = sum(oof[rule][i][3] for i in tasks)
        row = {'sr': float(pay.mean()), 'sr_ci': np.percentile(pay[idx].mean(1), [2.5, 97.5]).tolist(),
               'n_stops': len(stops), 'n_on_target': int(on),
               'median_stop_step': float(np.median(stops)) if stops else None,
               'mean_steps_used': float(used.mean()), 'params': chosen[rule]}
        for ref in ('full_budget', 'first_match'):
            rp = np.array([oof[ref][i][0] for i in tasks])
            ru = np.array([oof[ref][i][2] for i in tasks])
            row[f'dsr_vs_{ref}'] = float((pay - rp).mean())
            row[f'dsr_vs_{ref}_ci'] = np.percentile((pay - rp)[idx].mean(1), [2.5, 97.5]).tolist()
            row[f'dsteps_vs_{ref}'] = float((used - ru).mean())
            row[f'dsteps_vs_{ref}_ci'] = np.percentile((used - ru)[idx].mean(1), [2.5, 97.5]).tolist()
        res['rules'][rule] = row

    # M1 calibration quality (out-of-fold candidates).
    cr = np.array(calib_rows, dtype=float)
    calib = {}
    for name, mask in (('all', np.ones(len(cr), bool)), ('yes', cr[:, 2] > 0), ('no', cr[:, 3] > 0),
                       ('unverified', (cr[:, 2] == 0) & (cr[:, 3] == 0))):
        calib[name] = {'n': int(mask.sum()), 'mean_q': float(cr[mask, 0].mean()), 'frac_target': float(cr[mask, 1].mean())}
    res['m1_calibration'] = calib

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(args.out) / 'results.json', 'w'), ensure_ascii=False, indent=1)
    per = [{'task': i, **{f'{r}_pay': oof[r][i][0] for r in RULES}, **{f'{r}_stop': oof[r][i][1] for r in RULES},
            **{f'{r}_used': oof[r][i][2] for r in RULES}} for i in tasks]
    json.dump(per, open(Path(args.out) / 'per_episode.json', 'w'), indent=0)
    print(f"ever yes on target: {ever_yes_target}/100")
    print('M1 calibration (OOF):', json.dumps(calib, ensure_ascii=False))
    for rule, r in res['rules'].items():
        print(f"{rule:12s} SR={r['sr']:.3f} [{r['sr_ci'][0]:.3f},{r['sr_ci'][1]:.3f}] stops={r['n_stops']:3d} "
              f"on_t={r['n_on_target']:2d} med_stop={r['median_stop_step']} used={r['mean_steps_used']:.0f} | "
              f"vs full: dSR={r['dsr_vs_full_budget']:+.3f} [{r['dsr_vs_full_budget_ci'][0]:+.3f},{r['dsr_vs_full_budget_ci'][1]:+.3f}] "
              f"dsteps={r['dsteps_vs_full_budget']:+.0f} [{r['dsteps_vs_full_budget_ci'][0]:+.0f},{r['dsteps_vs_full_budget_ci'][1]:+.0f}]")


if __name__ == '__main__':
    main()
