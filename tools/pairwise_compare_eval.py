#!/usr/bin/env python
"""Offline test of pairwise (relative) candidate comparison with the navigation VLM.

Motivation (analysis_relative_evidence_20261104, analyze_yes_pool.py): after exploring to step
860, the goal is among the reachable "yes" candidates in 61/100 S6 episodes (median 3 such
candidates), but the calibrated absolute evidence q ranks it first in only 64% of them. A
perfect choice among them would lift the end-choice SR from 0.47 to 0.62. F6/F12 showed that
absolute single-candidate judgements cannot separate the goal from hard distractors. This
script asks whether a RELATIVE judgement can: the same model (qwen3.5-flash, temperature 0,
thinking off, as in navigation) sees two marked candidates and says which one matches the
description better. Each pair is asked in both orders to cancel position bias.

Unit: S6 episodes with >= 2 reachable yes candidates at the last moment before T_FB. The top 3
candidates by out-of-fold q are compared pairwise. Each candidate is shown in its best yes view
(largest box among its yes boxes), full frame with a red box, so surroundings stay visible.

Choice rules compared on the same episodes, scored by payoff-by-distance (F15 table):
  q_top         highest q (current evidence)
  pair_top2     switch from q's #1 to #2 only if #2 wins consistently in both orders
  tournament    round robin among the top 3: consistent win = 1, otherwise 0.5 each; q breaks ties
  votes         every single answer is one vote; q breaks ties

Ground truth is used only for scoring. Answers are cached in <out>/answers.jsonl.

Run (repository root):
  QWEN_MODEL_NAME=qwen3.5-flash PSGNAV_MODEL_BACKEND=openai PYTHONPATH=habitat-lab:. \
      $PSGNAV_PYTHON tools/pairwise_compare_eval.py [--dry-run]
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import tools.analyze_relative_evidence as R  # noqa: E402
from tools.analyze_yes_pool import load, oof_q, last_before  # noqa: E402
from tools.offline_verifier_eval import DiskCachedClient, _ask, _rf, _loads_object, draw_mark, load_rgb  # noqa: E402

T_FB = 860
TOP = 3
OUT = REPO / 'outputs' / 'analysis_pairwise_20261104'
CHOICES = ('A', 'B', 'same_object', 'unsure')
SCHEMA = {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': list(CHOICES)}},
          'required': ['choice'], 'additionalProperties': False}


def build_pair_prompt(category, intrinsic, extrinsic):
    return (
        'Stage: candidate comparison.\n'
        f'Two images are given. In image A and in image B, a red box (or red circle) marks one candidate {category}. '
        f'The task is to find one specific {category} in this house, described below. '
        'The two marked candidates may be different objects or, occasionally, the same object seen twice.\n'
        f'Description of the object itself:\n{intrinsic}\n'
        f'Description of its surroundings:\n{extrinsic}\n'
        'Compare the two marked candidates with each other against the description, using both the '
        "object's own appearance and its visible surroundings. Decide which marked candidate is more likely "
        f'to be the described {category}. Answer "same_object" only if both images clearly show the same '
        'physical object, and "unsure" only if the images give no basis to prefer either.\n'
        'Return only one JSON object {"choice": "A" | "B" | "same_object" | "unsure"}. No other text.'
    )


def parse_pair(answer):
    data = _loads_object(answer)
    if set(data) != {'choice'} or data['choice'] not in CHOICES:
        raise ValueError('expected {"choice": A|B|same_object|unsure}')
    return data['choice']


class BoxPool(R.Pool):
    """R.Pool that also keeps each cluster's boxes (same clustering, same cluster order)."""

    def add(self, b):
        g = b.get('est_gps')
        if not g:
            return
        n_before = len(self.c)
        target = None
        for c in self.c:
            if math.dist(c['seed'], g) <= R.RADIUS:
                target = c
                break
        super().add(b)
        c = target if target is not None else self.c[n_before]
        c.setdefault('boxes', []).append(b)


def pool_at(ep, t):
    boxes = sorted([b for b in ep['boxes'] if b.get('step') is not None and b.get('est_gps') and b['step'] <= t],
                   key=lambda b: b['step'])
    p = BoxPool()
    for b in boxes:
        p.add(b)
    return p.c


def best_view(ep, cluster, frame_image):
    yes = [b for b in cluster['boxes'] if (b.get('confirmation') or b.get('admission')) == 'yes']
    yes = [b for b in yes if frame_image.get(b['step'])]
    if not yes:
        return None
    b = max(yes, key=lambda x: (x.get('box_area_frac') or 0.0, x['step']))
    mark = ('box', b['box']) if b.get('box') else ('center', b.get('center_px'))
    if mark[1] is None:
        return None
    return {'image': frame_image[b['step']], 'mark': mark, 'step': b['step']}


def build_units(eps, M, END, Q, at_step=None):
    """at_step: {task: step} to build the comparison at that moment (e.g. where a stop rule
    fired); default is the last moment before T_FB (the end choice)."""
    units = []
    for i in sorted(M):
        if at_step is None:
            j = last_before(M[i], T_FB)
        else:
            s = at_step.get(i)
            j = next((k for k, m in enumerate(M[i]) if m['t'] == s), None) if s is not None else None
        if j is None:
            continue
        m, q = M[i][j], Q[i][j]
        ok = np.flatnonzero(m['yes'] & R.reachable(m))
        if len(ok) < 2:
            continue
        ep = eps[i]
        clusters = pool_at(ep, m['t'])
        assert len(clusters) == len(m['X']), (i, len(clusters), len(m['X']))
        frame_image = {f['step']: f['image'] for f in ep['frames'] if f.get('image')}
        order = ok[np.argsort(-q[ok])][:TOP]
        cands = []
        for k in order:
            v = best_view(ep, clusters[k], frame_image)
            if v is None:
                continue
            cands.append({'k': int(k), 'q': float(q[k]), 'tdist': float(m['tdist'][k]),
                          'payoff': R.payoff_of_distance(m['tdist'][k]), 'on': bool(m['tdist'][k] < 1.0),
                          'dagent': float(m['dagent'][k]), **v})
        if len(cands) < 2:
            continue
        units.append({'task': i, 't': m['t'], 'category': ep['category'],
                      'intrinsic': ep.get('intrinsic_attributes') or '',
                      'extrinsic': ep.get('extrinsic_attributes') or '', 'cands': cands})
    return units


def run_calls(units, root, workers):
    from utils.model_client import ModelClient
    OUT.mkdir(parents=True, exist_ok=True)
    cc = DiskCachedClient(ModelClient(), OUT / 'answers.jsonl')
    jobs = []
    for u in units:
        prompt = build_pair_prompt(u['category'], u['intrinsic'], u['extrinsic'])
        for a in range(len(u['cands'])):
            for b in range(a + 1, len(u['cands'])):
                for first, second in ((a, b), (b, a)):
                    jobs.append((u['task'], first, second, prompt))

    def one(job):
        task, x, y, prompt = job
        u = next(v for v in units if v['task'] == task)
        cx, cy = u['cands'][x], u['cands'][y]
        imgs = [draw_mark(load_rgb(root, cx['image']), cx['mark'], label='A'),
                draw_mark(load_rgb(root, cy['image']), cy['mark'], label='B')]
        try:
            ans = _ask(cc, 'pairwise_v1', prompt, imgs, parse_pair, _rf('pair_choice', SCHEMA), max_tokens=64)
        except Exception as e:  # noqa: BLE001 - record and continue
            ans = f'error:{type(e).__name__}'
        return (task, x, y, ans)

    answers = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            task, x, y, ans = f.result()
            answers[(task, x, y)] = ans
            if n % 50 == 0:
                print(f'[calls] {n}/{len(jobs)}', flush=True)
    print(f'[calls] network={cc.calls["network"]} cache={cc.calls["cache"]}', flush=True)
    return answers


def outcome(answers, task, x, y):
    """'x' / 'y' when consistent in both orders, else 'tie'."""
    a1 = answers.get((task, x, y))   # x shown as A
    a2 = answers.get((task, y, x))   # y shown as A
    if a1 == 'A' and a2 == 'B':
        return 'x'
    if a1 == 'B' and a2 == 'A':
        return 'y'
    return 'tie'


def choose(u, answers, rule):
    cands = u['cands']
    n = len(cands)
    if rule == 'q_top':
        return 0
    if rule == 'pair_top2':
        return 1 if outcome(answers, u['task'], 0, 1) == 'y' else 0
    if rule == 'tournament':
        pts = [0.0] * n
        for a in range(n):
            for b in range(a + 1, n):
                o = outcome(answers, u['task'], a, b)
                if o == 'x':
                    pts[a] += 1
                elif o == 'y':
                    pts[b] += 1
                else:
                    pts[a] += 0.5
                    pts[b] += 0.5
        return max(range(n), key=lambda k: (pts[k], -k))
    if rule == 'votes':
        v = [0] * n
        for (task, x, y), ans in answers.items():
            if task != u['task']:
                continue
            if ans == 'A':
                v[x] += 1
            elif ans == 'B':
                v[y] += 1
        return max(range(n), key=lambda k: (v[k], -k))
    raise ValueError(rule)


def evaluate(units, answers, eps, M, END, Q):
    rules = ('q_top', 'pair_top2', 'tournament', 'votes')
    res = {'n_units': len(units), 'n_calls': len(answers)}
    flat = Counter(a for a in answers.values())
    res['answer_dist'] = dict(flat)
    # Discrimination on mixed pairs (exactly one candidate on the goal).
    mixed = Counter()
    for u in units:
        c = u['cands']
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                if c[a]['on'] == c[b]['on']:
                    continue
                o = outcome(answers, u['task'], a, b)
                goal_is_x = c[a]['on']
                if o == 'tie':
                    mixed['tie'] += 1
                elif (o == 'x') == goal_is_x:
                    mixed['goal_wins'] += 1
                else:
                    mixed['distractor_wins'] += 1
    res['mixed_pairs'] = dict(mixed)
    # Order consistency and position bias.
    pairs = {(t, min(x, y), max(x, y)) for (t, x, y) in answers}
    cons = sum(outcome(answers, t, x, y) != 'tie' for (t, x, y) in pairs)
    res['consistent_frac'] = cons / max(1, len(pairs))
    res['first_shown_frac'] = flat.get('A', 0) / max(1, flat.get('A', 0) + flat.get('B', 0))
    # Choice quality on the units.
    for rule in rules:
        pay = [u['cands'][choose(u, answers, rule)]['payoff'] for u in units]
        on = [u['cands'][choose(u, answers, rule)]['on'] for u in units]
        res[rule] = {'mean_payoff': float(np.mean(pay)), 'on_target': int(sum(on))}
    res['oracle_top3'] = {'mean_payoff': float(np.mean([max(c['payoff'] for c in u['cands']) for u in units])),
                          'on_target': int(sum(any(c['on'] for c in u['cands']) for u in units))}
    # SR over all 100 episodes for the end choice at T_FB (episodes outside the units keep q_top).
    base = {}
    for i in sorted(M):
        j = last_before(M[i], T_FB)
        b = R.best_yes(M[i][j], Q[i][j]) if j is not None else None
        base[i] = R._stop(M[i][j], b)[0] if b is not None else 0.0
    by_task = {u['task']: u for u in units}
    rng = np.random.default_rng(20261104)
    idx = rng.integers(0, len(base), size=(4000, len(base)))
    tasks = sorted(base)
    ref = np.array([base[i] for i in tasks])
    for rule in rules:
        v = np.array([by_task[i]['cands'][choose(by_task[i], answers, rule)]['payoff'] if i in by_task else base[i]
                      for i in tasks])
        d = v - ref
        res[rule]['end_choice_sr_100'] = float(v.mean())
        res[rule]['dsr_vs_q_top'] = float(d.mean())
        res[rule]['dsr_ci'] = np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()
    return res


def evaluate_at_rule(units, answers, rule_oof):
    """SR / steps of a stop rule when its chosen candidate is replaced by the pairwise winner.

    rule_oof: {task: [payoff, stop_step, steps_used, on]} of the stop rule (out-of-fold).
    Episodes without a comparison unit keep the rule's own outcome.
    """
    by_task = {u['task']: u for u in units}
    tasks = sorted(rule_oof)
    rng = np.random.default_rng(20261104)
    idx = rng.integers(0, len(tasks), size=(4000, len(tasks)))
    base_pay = np.array([rule_oof[i][0] for i in tasks])
    base_used = np.array([rule_oof[i][2] for i in tasks])
    out = {'rule_alone': {'sr': float(base_pay.mean()), 'steps': float(base_used.mean())}}
    for rr in ('pair_top2', 'tournament', 'votes'):
        pay, used = [], []
        for i in tasks:
            u = by_task.get(i)
            if u is None:
                pay.append(rule_oof[i][0])
                used.append(rule_oof[i][2])
                continue
            c = u['cands'][choose(u, answers, rr)]
            pay.append(c['payoff'])
            used.append(u['t'] + R.WALK_A * c['dagent'])
        pay, used = np.array(pay), np.array(used)
        d = pay - base_pay
        out[rr] = {'sr': float(pay.mean()), 'steps': float(used.mean()), 'dsr_vs_rule': float(d.mean()),
                   'dsr_ci': np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--at-rule', default=None,
                    help='compare at the stop moments of this rule in ablation_oof.json instead of at T_FB')
    ap.add_argument('--oof', default=str(REPO / 'outputs/analysis_relative_evidence_20261104/ablation_oof.json'))
    args = ap.parse_args()
    eps, M, END, SC, NI = load()
    Q, _ = oof_q(M, SC)
    if args.at_rule:
        rule_oof = {int(k): v for k, v in json.load(open(args.oof))[args.at_rule].items()}
        at = {i: v[1] for i, v in rule_oof.items() if v[1] is not None}
        units = build_units(eps, M, END, Q, at_step=at)
        print(f'[at-rule] {args.at_rule}: {len(at)} stops, {len(units)} with >= 2 reachable yes candidates', flush=True)
        n_calls = sum(len(u['cands']) * (len(u['cands']) - 1) for u in units)
        print(f'[units] {n_calls} calls (cached ones are free)', flush=True)
        if args.dry_run:
            return
        root = REPO / 'data' / 'verification_set' / R.RUN
        answers = run_calls(units, root, args.workers)
        res = {'at_rule': args.at_rule, 'n_units': len(units), **evaluate_at_rule(units, answers, rule_oof)}
        mixed = Counter()
        for u in units:
            c = u['cands']
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    if c[a]['on'] != c[b]['on']:
                        o = outcome(answers, u['task'], a, b)
                        mixed['tie' if o == 'tie' else ('goal_wins' if (o == 'x') == c[a]['on'] else 'distractor_wins')] += 1
        res['mixed_pairs'] = dict(mixed)
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT / f'results_at_{args.at_rule}.json', 'w'), ensure_ascii=False, indent=1)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    units = build_units(eps, M, END, Q)
    n_calls = sum(len(u['cands']) * (len(u['cands']) - 1) for u in units)
    n_goal = sum(any(c['on'] for c in u['cands']) for u in units)
    n_qtop_on = sum(u['cands'][0]['on'] for u in units)
    print(f'[units] {len(units)} episodes, {n_calls} calls; goal in top-{TOP}: {n_goal}; q_top on goal: {n_qtop_on}',
          flush=True)
    if args.dry_run:
        return
    root = REPO / 'data' / 'verification_set' / R.RUN
    answers = run_calls(units, root, args.workers)
    res = evaluate(units, answers, eps, M, END, Q)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
    json.dump([{**{k: v for k, v in u.items() if k != 'cands'},
                'cands': [{k: v for k, v in c.items() if k != 'mark'} for c in u['cands']],
                'answers': {f'{x}>{y}': a for (t, x, y), a in answers.items() if t == u['task']}} for u in units],
              open(OUT / 'units.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
