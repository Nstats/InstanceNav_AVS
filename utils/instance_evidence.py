"""Episode-local candidates and verification evidence without annotation identities."""
from dataclasses import dataclass, field
from io import BytesIO
import hashlib

import numpy as np
from PIL import Image

from utils.instance_verifier import _candidate_image


def _iou(first, second):
    first, second = np.asarray(first), np.asarray(second)
    size = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2])) - intersection
    return intersection / union if union > 0 else 0.


@dataclass
class CandidateEvidence:
    track_id: int
    node_id: object
    gps: object
    box: list
    step: int
    current: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    positive_steps: set = field(default_factory=set)
    # (map row, map column, policy step) for each increment in the existing counter.
    votes: list = field(default_factory=list)
    rejected_step: int = -1
    last_check_key: object = None
    last_check_result: object = None
    detection: object = None
    vetoed: bool = False

    def record_positive(self, rgb, step, box):
        # A repeated detector call for the same observation does not create another view.
        if step not in self.frames:
            image = _candidate_image(rgb, [box])
            with BytesIO() as buffer:
                image.save(buffer, format='PNG', compress_level=3)
                png = buffer.getvalue()
            self.frames[step] = {
                'png': png, 'step': step, 'box': list(map(float, box)),
                'sha256': hashlib.sha256(png).hexdigest(),
            }

    def images_and_metadata(self):
        images, metadata = [], []
        for frame in self.frames.values():
            with Image.open(BytesIO(frame['png'])) as image:
                images.append(image.convert('RGB'))
            metadata.append({k: v for k, v in frame.items() if k != 'png'})
        return images, metadata


class PositiveEvidenceStore:
    def __init__(self, merge_distance):
        self.merge_distance = merge_distance
        self.tracks = []

    def associate(self, candidates, step):
        """One-to-one spatial association; graph IDs take precedence over geometry.

        Same-frame boxes must overlap to reuse a track. Different graph nodes never
        merge. Equidistant spatial matches start separate tracks rather than mixing
        their evidence. This is geometric tracking, not oracle instance identity.
        """
        assigned, used = {}, set()
        for index, candidate in enumerate(candidates):
            if candidate['node_id'] is None:
                continue
            for track in self.tracks:
                if not track.vetoed and track.node_id == candidate['node_id']:
                    assigned[index] = track
                    used.add(track.track_id)
                    break

        pairs = []
        for index, candidate in enumerate(candidates):
            if index in assigned or candidate['node_id'] is not None or candidate['gps'] is None:
                continue
            options = []
            for track in self.tracks:
                if track.vetoed or track.node_id is not None or track.gps is None or track.track_id in used:
                    continue
                delta = np.abs(candidate['gps'] - track.gps)
                if np.max(delta) > self.merge_distance:
                    continue
                if track.step == step and _iou(candidate['box'], track.box) < .5:
                    continue
                options.append((float(np.linalg.norm(delta)), track.track_id))
            options.sort()
            if len(options) > 1 and options[1][0] - options[0][0] < .05:
                continue
            if options:
                pairs.append((options[0][0], index, options[0][1]))
        # A verified box has precedence over an unverified approach candidate.
        # Distance still decides association within each group.
        for _, index, track_id in sorted(
                pairs, key=lambda pair: (not candidates[pair[1]].get('lock_passed', False), *pair)):
            if track_id not in used:
                assigned[index] = self.tracks[track_id]
                used.add(track_id)

        for index, candidate in enumerate(candidates):
            if index not in assigned:
                track = CandidateEvidence(len(self.tracks), candidate['node_id'], candidate['gps'],
                                          candidate['box'], step)
                self.tracks.append(track)
                assigned[index] = track
            track = assigned[index]
            track.gps = candidate['gps']
            track.box = candidate['box']
            track.step = step
            track.current = candidate
        return [assigned[index] for index in range(len(candidates))]

    def reset_progress(self):
        # Recovery clears counts, but all earlier positive observations remain available.
        for track in self.tracks:
            track.votes.clear()


class RecoveryEvidenceGuard:
    """Navigation eligibility only; never removes semantic observations or votes."""

    # Match the navigator's existing movement threshold. Repeated failures need
    # a different approach position, not merely another turn or an elapsed timer.
    MIN_PROGRESS_M = .05
    SAME_APPROACH_M = .25
    RETRY_CLEARANCE_M = .5

    def __init__(self):
        self.origin = None
        self.waiting_for_progress = False
        self.step = -1
        self.failed_approaches = {}
        self.avoid_positions = {}

    def start(self, track_id, position, step):
        # Re-entering recovery before making progress must not move the origin.
        if not self.waiting_for_progress:
            self.origin = position.copy() if position is not None else None
        self.waiting_for_progress = True
        self.step = step
        if track_id is not None and position is not None:
            previous = self.failed_approaches.setdefault(track_id, [])
            if any(np.linalg.norm(position - old) <= self.SAME_APPROACH_M for old in previous):
                self.avoid_positions.setdefault(track_id, []).append(position.copy())
            previous.append(position.copy())

    def blocked_reason(self, track_id, position, step, *, recovering, stuck):
        if self.step < 0:
            return None
        if step <= self.step:
            return 'needs_new_observation'
        if recovering or stuck:
            return 'recovery_in_progress'
        if position is None:
            return 'missing_agent_position'
        if self.waiting_for_progress:
            if self.origin is None:
                self.origin = position.copy()
                return 'no_observed_movement'
            if np.linalg.norm(position - self.origin) < self.MIN_PROGRESS_M:
                return 'no_observed_movement'
            self.origin = None
            self.waiting_for_progress = False
        if any(np.linalg.norm(position - old) < self.RETRY_CLEARANCE_M
               for old in self.avoid_positions.get(track_id, [])):
            return 'same_failed_approach'
        return None
