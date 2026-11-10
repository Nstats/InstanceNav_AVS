#!/usr/bin/env python
"""Where is the headroom? Decompose the S6 stop problem into "when" and "which".

Uses the same full candidate pool, payoff-by-distance and reachability as
analyze_relative_evidence.py. Ground truth is used for the oracle ceilings only.

Ceilings (all on reachable candidates, payoff by distance to the goal):
  oracle_when    stop at the first moment the goal is a reachable yes candidate
  oracle_which   explore to T_fb, then pick the best yes candidate with perfect knowledge
  model_which    explore to T_fb, then pick the yes candidate with the highest calibrated q
                 (scene-grouped out-of-fold, as in analyze_relative_evidence.py)
plus the composition of the yes set at T_fb: how many yes candidates, how many of them sit
on the goal, and how the goal ranks under q.

Also a "veto" family: at the FIRST yes event, accept only if the number of rejected
candidates is >= k (F13); otherwise continue and pick the best yes candidate at T_fb.

Run: PYTHONPATH=habitat-lab:. $PSGNAV_PYTHON tools/analyze_yes_pool.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import tools.analyze_relative_evidence as R  # noqa: E402
from tools.analyze_verification_set import load_set  # noqa: E402

RNG = np.random.default_rng(20261104)


def load():
    data = load_set(REPO / 'data' / 'verification_set', REPO / 'data', runs=[R.RUN])
    eps = data[R.RUN]['episodes']
    ents = json.load(gzip.open(REPO / 'data/datasets/instancenav/val10/val10.json.gz'))['episodes']
    ds = {(e['scene_id'].split('/')[-1], str(e['episode_id']), str(e['goal_object_id'])): e for e in ents}
    M, END, SC, NI = {}, {}, {}, {}
    for i, e in sorted(eps.items()):
        M[i], END[i] = R.episode_moments(e, ds[tuple(e['key'])])
        SC[i] = e['key'][0]
        NI[i] = len(e['same_category_instances'] or {})
    return eps, M, END, SC, NI


def oof_q(M, SC):
    tasks = sorted(M)
    scenes = sorted(set(SC.values()))
    rng = np.random.default_rng(20261104)
    rng.shuffle(scenes)
    fold = {s: k % 5 for k, s in enumerate(scenes)}
    Q = {}
    for f in range(5):
        Xs, ys = [], []
        for i in tasks:
            if fold[SC[i]] == f:
                continue
            for m in M[i]:
                if m['t'] % 50 == 0:
                    Xs.append(m['X'])
                    ys.append((m['tdist'] < 1.0).astype(float))
        model = R.fit_logit(np.vstack(Xs), np.concatenate(ys))
        for i in tasks:
            if fold[SC[i]] == f:
                Q[i] = [R.predict(model, m['X']) for m in M[i]]
    return Q, fold


def last_before(moments, T):
    k = None
    for j, m in enumerate(moments):
        if m['t'] >= T:
            break
        k = j
    return k


def main():
    eps, M, END, SC, NI = load()
    Q, fold = oof_q(M, SC)
    tasks = sorted(M)
    out = {}
    for T in (500, 700, 860):
        rows = []
        for i in tasks:
            j = last_before(M[i], T)
            if j is None:
                rows.append({'task': i, 'n_yes': 0, 'goal_yes': False, 'oracle': 0.0, 'model': 0.0, 'rank': None})
                continue
            m, q = M[i][j], Q[i][j]
            ok = m['yes'] & R.reachable(m)
            idx = np.flatnonzero(ok)
            if not len(idx):
                rows.append({'task': i, 'n_yes': 0, 'goal_yes': False, 'oracle': 0.0, 'model': 0.0, 'rank': None})
                continue
            pays = np.array([R.payoff_of_distance(m['tdist'][k]) for k in idx])
            on = m['tdist'][idx] < 1.0
            qq = q[idx]
            order = np.argsort(-qq)
            rank = int(np.flatnonzero(on[order])[0]) + 1 if on.any() else None
            rows.append({'task': i, 'n_yes': int(len(idx)), 'n_on': int(on.sum()), 'goal_yes': bool(on.any()),
                         'oracle': float(pays.max()), 'model': float(pays[np.argmax(qq)]), 'rank': rank,
                         'n_inst': NI[i]})
        g = [r for r in rows if r['goal_yes']]
        out[T] = {
            'oracle_which_sr': float(np.mean([r['oracle'] for r in rows])),
            'model_which_sr': float(np.mean([r['model'] for r in rows])),
            'n_goal_yes': len(g),
            'n_yes_when_goal_yes': np.percentile([r['n_yes'] for r in g], [25, 50, 75, 90]).tolist() if g else None,
            'goal_rank1_frac': float(np.mean([r['rank'] == 1 for r in g])) if g else None,
            'goal_rank_dist': {str(k): sum(r['rank'] == k for r in g) for k in range(1, 6)},
            'goal_rank_gt5': sum((r['rank'] or 0) > 5 for r in g),
            'n_yes_2to3_frac': float(np.mean([2 <= r['n_yes'] <= 3 for r in g])) if g else None,
            'n_yes_1_frac': float(np.mean([r['n_yes'] == 1 for r in g])) if g else None,
        }
        print(f"T_fb={T}: oracle_which={out[T]['oracle_which_sr']:.3f} model_which={out[T]['model_which_sr']:.3f} "
              f"goal among reachable yes in {len(g)} tasks; n_yes p25/50/75/90={out[T]['n_yes_when_goal_yes']}; "
              f"goal ranked 1st by q: {out[T]['goal_rank1_frac']:.2f}; ranks {out[T]['goal_rank_dist']} >5:{out[T]['goal_rank_gt5']}",
              flush=True)

    # oracle_when: first moment the goal is a reachable yes candidate.
    when, when_step = [], []
    for i in tasks:
        hit = None
        for m in M[i]:
            ok = m['yes'] & R.reachable(m) & (m['tdist'] < 1.0)
            if ok.any():
                hit = m
                break
        when.append(0.89 if hit is not None else 0.0)
        if hit is not None:
            when_step.append(hit['t'])
    out['oracle_when'] = {'sr_lower_bound': float(np.mean(when)), 'n': len(when_step),
                          'median_step': float(np.median(when_step)) if when_step else None}
    print(f"oracle_when: goal reachable-yes in {len(when_step)} tasks, first at median step "
          f"{out['oracle_when']['median_step']}; SR >= {out['oracle_when']['sr_lower_bound']:.3f} (others counted 0)")

    # Veto family: first yes accepted only if n_rej >= k; otherwise go on and pick best q at T_fb.
    veto = {}
    for k in (0, 1, 2, 3, 4, 6):
        for T in (700, 860):
            pays, steps = [], []
            for i in tasks:
                res = None
                for m, q in zip(M[i], Q[i]):
                    if m['t'] >= T:
                        break
                    if m['yes'].any():
                        if m['n_rej'] >= k:
                            b = R.best_yes(m, q)
                            if b is not None:
                                res = R._stop(m, b)
                        break
                if res is None:
                    j = last_before(M[i], T)
                    b = R.best_yes(M[i][j], Q[i][j]) if j is not None else None
                    res = R._stop(M[i][j], b) if b is not None else (0.0, None, END[i], False)
                pays.append(res[0])
                steps.append(res[2])
            veto[(k, T)] = (float(np.mean(pays)), float(np.mean(steps)))
            print(f"veto k={k} T_fb={T}: SR={veto[(k, T)][0]:.3f} mean steps used={veto[(k, T)][1]:.0f}", flush=True)
    out['veto'] = {f'k{k}_T{T}': v for (k, T), v in veto.items()}

    od = REPO / 'outputs' / 'analysis_relative_evidence_20261104'
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / 'yes_pool.json', 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
