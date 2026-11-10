#!/usr/bin/env python
"""Offline A/B of the instance-verification prompt on stored collect-run frames.

Question: does redefining "no" (clear contradiction only; unclear goes to "unknown") change the
main AVS flow? Only a changed "yes" can change it, because no and unknown act identically there.

Arms, all asked on the same stored frame with the same boxes:
    old_a, old_b  the production prompt, asked twice (noise floor; API cache disabled)
    new           the same request with one sentence replaced (NEW_RULE); the "yes" wording is untouched
Both go through utils.instance_verifier.verify_candidate_decisions, so stage prefix, schema and
retries match production.

Phases (a shuffled episode order keeps a deadline-truncated run unbiased):
    strat  random groups per class, for unbiased verdict rates per class
    walk   each episode's groups in time order until every arm has a yes (first-lock replay)

Run: PSGNAV_ENV_FILE=.env PSGNAV_API_CACHE_SIZE=0 PSGNAV_API_CONCURRENCY=8 \
     python tools/offline_prompt_ab.py --out outputs/prompt_ab_20261106 --deadline 3300
Then: python tools/offline_prompt_ab.py --out outputs/prompt_ab_20261106 --report
"""
import argparse
import glob
import json
import random
import sys
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROOT = REPO / 'data' / 'verification_set'
COLLECT = '20261103_011329_142191'
CLOSED_LOOP = ('20261030_211522_2605603', '20261031_133130_3729343', '20261101_145827_2973537')
OLD_RULE = 'If evidence is inadequate, answer unknown. '
NEW_RULE = ('Answer "no" only if the visible evidence clearly contradicts the description, so that this '
            'is definitely not the described object. If the object is partly hidden, too small or '
            'unclear, if the described attributes or surrounding context are not visible in this view, '
            'or if you cannot tell for any other reason, answer "unknown", not "no". ')
ARMS = ('old_a', 'old_b', 'new')
P_ON_TARGET, P_WRONG = 0.89, 0.22  # success given the lock outcome (F15)


class RewriteClient:
    """Pass-through client that swaps the rule sentence, failing loudly if it is absent."""

    def __init__(self, client):
        self.client = client

    def complete(self, prompt, image=None, **kwargs):
        if OLD_RULE not in prompt:
            raise RuntimeError('production prompt no longer contains the rule sentence')
        return self.client.complete(prompt.replace(OLD_RULE, NEW_RULE), image, **kwargs)


def load_episodes():
    return [json.load(open(p)) for p in sorted(glob.glob(str(ROOT / COLLECT / 'ep_*.json')))]


def hard_instances():
    hard = set()
    for run in CLOSED_LOOP:
        for path in glob.glob(str(ROOT / run / 'ep_*.json')):
            ep = json.load(open(path))
            for lock in ep.get('locks') or []:
                if lock.get('nearest_instance') is not None and lock.get('nearest_is_target') is False:
                    hard.add((ep['dataset_index'], lock['nearest_instance']))
    return hard


def box_class(ep, box, hard):
    if box.get('gt_instance') == ep['target_instance']:
        return 'target'
    if box.get('label') == 'same_category':
        return 'hard' if (ep['dataset_index'], box.get('gt_instance')) in hard else 'same'
    return 'other'


def groups_of(ep, hard):
    """One group = one production verification request: same frame, same boxes, same order."""
    grouped = OrderedDict()
    boxes = [b for b in ep['boxes'] if b.get('confirmation') in ('yes', 'no', 'unknown')]
    for b in sorted(boxes, key=lambda x: (x['step'], x['box_index'])):
        grouped.setdefault((b['step'], b['group_verification_id']), []).append(b)
    images = {f['step']: f['image'] for f in ep['frames'] if f.get('image')}
    out = []
    for (step, gid), members in grouped.items():
        if step not in images:
            continue
        out.append({'ep': ep['dataset_index'], 'step': step, 'gid': gid, 'image': images[step],
                    'boxes': [b['box'] for b in members],
                    'cls': [box_class(ep, b, hard) for b in members],
                    'depth': [b.get('top_depth_m') for b in members],
                    'logged': [b['confirmation'] for b in members]})
    return out


class Runner:
    def __init__(self, out, deadline):
        from utils.instance_verifier import verify_candidate_decisions
        from utils.model_client import ModelClient
        self.verify = verify_candidate_decisions
        self.old = ModelClient()
        self.new = RewriteClient(self.old)
        self.path = out / 'answers.jsonl'
        self.lock = threading.Lock()
        self.stop_at = time.time() + deadline
        self.done = {}
        if self.path.exists():
            for line in open(self.path):
                rec = json.loads(line)
                self.done[(rec['phase'], rec['ep'], rec['step'], rec['gid'])] = rec

    def expired(self):
        return time.time() >= self.stop_at

    def ask(self, phase, g, ep):
        key = (phase, g['ep'], g['step'], g['gid'])
        if key in self.done:
            return self.done[key]
        import numpy as np
        from PIL import Image
        rgb = np.asarray(Image.open(ROOT / COLLECT / g['image']).convert('RGB'))
        attrs = {k: ep[k] for k in ('intrinsic_attributes', 'extrinsic_attributes')}
        rec = {**g, 'phase': phase}
        context = {'stage': 'goal_confirmation', 'episode': g['ep'], 'step': g['step']}
        for arm in ARMS:
            client = self.new if arm == 'new' else self.old
            try:
                rec[arm] = self.verify(client, rgb, g['boxes'], ep['category'], attrs, log_context=dict(context))
            except Exception as error:  # a failed request must not stop the run; it is reported
                rec[arm] = None
                rec['error'] = f'{type(error).__name__}: {error}'[:200]
        with self.lock:
            self.done[key] = rec
            with open(self.path, 'a') as stream:
                stream.write(json.dumps(rec) + '\n')
        return rec

    def walk(self, ep, groups, cap):
        seen = set()
        for g in groups[:cap]:
            if self.expired():
                return False
            rec = self.ask('walk', g, ep)
            seen.update(arm for arm in ARMS if rec.get(arm) and 'yes' in rec[arm])
            if len(seen) == len(ARMS):
                break
        with self.lock, open(self.path.with_name('walk_done.jsonl'), 'a') as stream:
            stream.write(json.dumps({'ep': ep['dataset_index']}) + '\n')
        return True


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prompt.json').write_text(json.dumps({'old_rule': OLD_RULE, 'new_rule': NEW_RULE}, indent=1))
    hard, eps = hard_instances(), load_episodes()
    by_ep = {ep['dataset_index']: ep for ep in eps}
    groups = {ep['dataset_index']: groups_of(ep, hard) for ep in eps}
    rng = random.Random(args.seed)
    pools = defaultdict(list)
    for gs in groups.values():
        for g in gs:
            pools[next((c for c in ('target', 'hard', 'same', 'other') if c in g['cls']))].append(g)
    sample = []
    for name in sorted(pools):
        rng.shuffle(pools[name])
        sample += pools[name][:args.per_class]
    rng.shuffle(sample)
    order = list(groups)
    rng.shuffle(order)
    print('pools', {k: len(v) for k, v in pools.items()}, 'strat', len(sample), flush=True)
    runner = Runner(out, args.deadline)
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda g: None if runner.expired() else runner.ask('strat', g, by_ep[g['ep']]), sample))
        print(f'strat done in {time.time() - start:.0f} s', flush=True)
        finished = list(pool.map(lambda i: None if runner.expired() else runner.walk(by_ep[i], groups[i], args.cap), order))
    print(f'walk episodes finished {sum(bool(x) for x in finished)}/{len(order)}; '
          f'elapsed {time.time() - start:.0f} s; groups {len(runner.done)}', flush=True)


def load_answers(out):
    recs = [json.loads(line) for line in open(out / 'answers.jsonl')]
    return [r for r in recs if all(r.get(a) is not None for a in ARMS)], len(recs)


def rate_table(recs):
    print('\nVerdict rates per box (strat phase, unbiased within class)')
    print(f'{"class":8s}{"n":>6s}' + ''.join(f'{a + " " + v:>14s}' for a in ('old_a', 'new') for v in ('yes', 'no', 'unk')))
    for cls in ('target', 'hard', 'same', 'other'):
        c = {a: Counter() for a in ARMS}
        for r in recs:
            for i, k in enumerate(r['cls']):
                if k == cls:
                    for a in ARMS:
                        c[a][r[a][i]] += 1
        n = sum(c['old_a'].values())
        if n:
            print(f'{cls:8s}{n:6d}' + ''.join(f'{c[a][v] / n:14.3f}' for a in ('old_a', 'new') for v in ('yes', 'no', 'unknown')))


def flip_table(recs):
    print('\n"yes" flips per box: noise floor vs prompt change')
    print(f'{"class":8s}{"n":>6s}{"old_a/old_b differ":>20s}{"old_a/new differ":>18s}{"yes->not (new)":>16s}{"not->yes (new)":>16s}')
    for cls in ('all', 'target', 'hard', 'same', 'other'):
        n = noise = change = lost = gained = 0
        for r in recs:
            for i, k in enumerate(r['cls']):
                if cls in ('all', k):
                    a, b, w = r['old_a'][i] == 'yes', r['old_b'][i] == 'yes', r['new'][i] == 'yes'
                    n += 1
                    noise += a != b
                    change += a != w
                    lost += a and not w
                    gained += w and not a
        if n:
            print(f'{cls:8s}{n:6d}{noise / n:20.3f}{change / n:18.3f}{lost / n:16.3f}{gained / n:16.3f}')


def first_lock(recs, arm):
    for r in sorted(recs, key=lambda x: x['step']):
        yes = [i for i, v in enumerate(r[arm]) if v == 'yes']
        if yes:
            best = min(yes, key=lambda i: r['depth'][i] if r['depth'][i] is not None else 99)
            return r['step'], r['cls'][best]
    return None


def replay(recs, done):
    per_ep = defaultdict(list)
    for r in recs:
        if r['ep'] in done:
            per_ep[r['ep']].append(r)
    print(f'\nFirst-lock replay on {len(per_ep)} fully walked episodes')
    print(f'{"arm":8s}{"on target":>10s}{"wrong":>7s}{"no lock":>9s}{"est. SR":>9s}{"median step":>13s}')
    locks = {}
    for arm in ARMS:
        res = {ep: first_lock(rs, arm) for ep, rs in per_ep.items()}
        locks[arm] = res
        hit = sum(1 for v in res.values() if v and v[1] == 'target')
        wrong = sum(1 for v in res.values() if v and v[1] != 'target')
        steps = sorted(v[0] for v in res.values() if v)
        n = max(len(res), 1)
        print(f'{arm:8s}{hit:10d}{wrong:7d}{n - hit - wrong:9d}'
              f'{100 * (P_ON_TARGET * hit + P_WRONG * wrong) / n:9.1f}{steps[len(steps) // 2] if steps else 0:13d}')
    for a, b in (('old_a', 'old_b'), ('old_a', 'new'), ('old_b', 'new')):
        same = sum(locks[a][e] == locks[b][e] for e in per_ep)
        outcome = [((locks[a][e] or (0, None))[1] == 'target', (locks[b][e] or (0, None))[1] == 'target') for e in per_ep]
        print(f'  {a} vs {b}: same first lock {same}/{len(per_ep)}; '
              f'on-target lost {sum(x and not y for x, y in outcome)}, gained {sum(y and not x for x, y in outcome)}')


def report(args):
    out = Path(args.out)
    recs, total = load_answers(out)
    print(f'groups {total}, complete {len(recs)}, failed {total - len(recs)}')
    agree = [(r['logged'][i] == 'yes') == (r['old_a'][i] == 'yes') for r in recs for i in range(len(r['cls']))]
    print(f'logged vs fresh old_a same yes/not-yes: {sum(agree) / max(len(agree), 1):.3f} (n={len(agree)})')
    rate_table([r for r in recs if r['phase'] == 'strat'])
    flip_table([r for r in recs if r['phase'] == 'strat'])
    done_path = out / 'walk_done.jsonl'
    done = {json.loads(line)['ep'] for line in open(done_path)} if done_path.exists() else set()
    replay([r for r in recs if r['phase'] == 'walk'], done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--deadline', type=float, default=3300, help='seconds of API time')
    ap.add_argument('--per-class', type=int, default=150)
    ap.add_argument('--cap', type=int, default=60, help='groups walked per episode')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=77)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    (report if args.report else run)(args)


if __name__ == '__main__':
    main()
