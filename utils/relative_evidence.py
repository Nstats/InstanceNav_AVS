"""Agent-side candidate pooling and frozen relative-evidence decisions.

This module never loads evaluation labels, simulator geometry or fitting tools.
Clustering/features deliberately match the corrected S6 offline replay.
"""
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

FEATURES = ['is_yes', 'is_no', 'log1p_yes', 'log1p_no', 'log_min_distance', 'log1p_sightings']
RADIUS = 1.0
SIGHT_GAP = 10
FALLBACK_STEP = 860


def finite_position(value):
    if value is None:
        return None
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return value if value.shape == (2,) and np.isfinite(value).all() else None


@dataclass
class Candidate:
    candidate_id: int
    seed: list
    sx: float = 0.0
    sy: float = 0.0
    n: int = 0
    yes: int = 0
    no: int = 0
    dmin: float = 99.0
    yes_gps: object = None
    sightings: int = 0
    last_step: object = None
    best_view: object = None

    @property
    def state(self):
        return 'yes' if self.yes else ('no' if self.no else 'unverified')

    @property
    def position(self):
        return list(self.yes_gps) if self.yes_gps is not None else [self.sx / self.n, self.sy / self.n]

    def features(self):
        return [float(self.state == 'yes'), float(self.state == 'no'),
                math.log1p(self.yes), math.log1p(self.no),
                math.log(max(0.5, self.dmin)), math.log1p(self.sightings)]

    def remember_view(self, rgb, box, step):
        if rgb is None or box is None:
            return
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return
        height, width = rgb.shape[:2]
        area = max(0., box[2] - box[0]) * max(0., box[3] - box[1]) / (width * height)
        if area <= 0 or (self.best_view and (area, step) < (self.best_view['area'], self.best_view['step'])):
            return
        image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).convert('RGB')
        with BytesIO() as buf:
            image.save(buf, format='PNG', compress_level=3)
            png = buf.getvalue()
        self.best_view = {'png': png, 'box': box.tolist(), 'step': step, 'area': area,
                          'sha256': hashlib.sha256(png).hexdigest()}

    def summary(self):
        return {'candidate_id': self.candidate_id, 'seed': self.seed, 'gps': self.position,
                'state': self.state, 'yes': self.yes, 'no': self.no, 'observations': self.n,
                'features': self.features(),
                'best_view': ({k: v for k, v in self.best_view.items() if k != 'png'}
                              if self.best_view else None)}


class CandidatePool:
    def __init__(self):
        self.candidates = []
        self.seen_events = set()

    def add(self, *, event_id, step, gps, answer=None, agent_position=None,
            observed_distance=None, rgb=None, box=None):
        if event_id in self.seen_events:
            return None
        self.seen_events.add(event_id)
        gps = finite_position(gps)
        if gps is None:
            return None
        if answer not in (None, 'yes', 'no', 'unknown'):
            raise ValueError(f'Invalid candidate answer: {answer}')
        candidate = next((c for c in self.candidates if math.dist(c.seed, gps) <= RADIUS), None)
        if candidate is None:
            candidate = Candidate(len(self.candidates), gps.tolist())
            self.candidates.append(candidate)
        if candidate.last_step is not None and step < candidate.last_step:
            raise ValueError('Candidate observations must be ordered by policy step')
        candidate.sx += float(gps[0])
        candidate.sy += float(gps[1])
        candidate.n += 1
        if answer == 'yes':
            candidate.yes += 1
            candidate.yes_gps = gps.tolist()
            candidate.remember_view(rgb, box, step)
        elif answer == 'no':
            candidate.no += 1
        if candidate.last_step is None or step - candidate.last_step >= SIGHT_GAP:
            candidate.sightings += 1
        candidate.last_step = step
        position = finite_position(agent_position)
        distance = math.dist(position, gps) if position is not None else observed_distance
        if distance is not None and math.isfinite(float(distance)) and float(distance) >= 0:
            candidate.dmin = min(candidate.dmin, float(distance))
        return candidate


class Calibration:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        if self.data.get('schema_version') != 1 or self.data.get('features') != FEATURES:
            raise ValueError('Unsupported relative-evidence calibration schema/features')
        if self.data.get('fallback_policy_step') != FALLBACK_STEP:
            raise ValueError('Calibration must use fallback policy step 860')
        if len(self.data.get('folds', [])) != 5:
            raise ValueError('Calibration must contain five scene-held-out models')
        for model in [*self.data['folds'], self.data['global']]:
            for name in ('mean', 'scale', 'coef'):
                values = np.asarray(model[name], dtype=float)
                if values.shape != (6,) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid calibration {name}')
            if np.any(np.asarray(model['scale']) <= 0) or not math.isfinite(model['intercept']):
                raise ValueError('Invalid calibration scaling/intercept')
            p = model['params']
            if not (math.isfinite(p['alpha']) and p['alpha'] >= 0
                    and math.isfinite(p['c']) and p['c'] >= 0 and 0 < p['tau'] <= 1):
                raise ValueError('Invalid relative-evidence stopping parameters')
        if any(type(f) is not int or not 0 <= f < 5 for f in self.data['scene_folds'].values()):
            raise ValueError('Invalid scene fold mapping')

    def for_scene(self, scene):
        name = Path(scene).name
        if not name.endswith('.basis.glb'):
            name = name.removesuffix('.glb') + '.basis.glb'
        fold = self.data['scene_folds'].get(name)
        return (self.data['global'], 'global') if fold is None else (self.data['folds'][fold], f'fold_{fold}')

    def metadata(self):
        return {'calibration_sha256': self.sha256, 'source': self.data['source'],
                'fold_params': [m['params'] for m in self.data['folds']],
                'global_params': self.data['global']['params'], 'fallback_policy_step': FALLBACK_STEP}

    @staticmethod
    def predict(model, features):
        x = np.asarray(features, dtype=float).reshape(-1, 6)
        z = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
        logits = z @ np.asarray(model['coef']) + model['intercept']
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def decision_snapshot(pool, model, *, step, position, remaining, use_u=True,
                      early_stop=True, drop_rejected=True):
    """Pure rule. An empty pool terminates only at fallback, never at startup."""
    candidates = pool.candidates
    q = Calibration.predict(model, [c.features() for c in candidates])
    alive = [i for i, c in enumerate(candidates) if not drop_rejected or c.state != 'no']
    leader = max(alive, key=lambda i: (q[i], -i)) if alive else None
    position = finite_position(position)
    distances = [math.dist(position, c.position) if position is not None else math.inf for c in candidates]
    reachable = [math.isfinite(d) and remaining >= math.ceil(8 * d + 10) for d in distances]
    ranked = sorted((i for i, c in enumerate(candidates) if c.state == 'yes' and reachable[i]),
                    key=lambda i: (-q[i], i))
    p = model['params']
    u = max(0., p['c'] - len(candidates)) if use_u else 0.
    denominator = float(sum(q[i] for i in alive)) + p['alpha'] * u
    p1 = float(q[leader] / denominator) if leader is not None and denominator > 0 else 0.
    reason = None
    if step >= FALLBACK_STEP:
        reason = 'fallback'
    elif (early_stop and leader is not None and candidates[leader].state == 'yes'
          and reachable[leader] and p1 >= p['tau']):
        reason = 'relative_threshold'
    return {'step': step, 'remaining_atomic': remaining, 'n_seen': len(candidates),
            'n_rejected': sum(c.state == 'no' for c in candidates), 'leader': leader,
            'q': q.tolist(), 'u': u, 'p1': p1, 'params': dict(p), 'reachable': reachable,
            'ranked_yes': ranked, 'trigger': reason, 'shortlist': ranked[:3] if reason else []}
