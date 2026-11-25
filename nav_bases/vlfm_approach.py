"""VLFM candidate approach (B/E5/RA). Only committed clouds enter the target map."""
import numpy as np

from nav_bases.avs_decision import AVSDecision
from utils.instance_evidence import PositiveEvidenceStore, _iou


class ApproachDecision(AVSDecision):
    def __init__(self, verify, *, admission=False, threshold=1, distance_threshold=4.0,
                 same_instance_m=.8, log=None):
        if threshold < 1 or int(threshold) != threshold:
            raise ValueError('threshold must be a positive integer')
        self.admission = admission
        self.threshold = int(threshold)
        super().__init__(verify, distance_threshold, same_instance_m, log)

    def reset(self):
        super().reset()
        self._evidence = PositiveEvidenceStore(self.same_instance_m)
        self.approach_goal = None
        self._approach_track_id = None

    def release_approach(self, reason, context=None):
        if reason == 'stuck':
            for track in self._evidence.tracks:
                track.positive_steps.clear()
        if self.approach_goal is not None:
            self._log('vlfm_approach_end', context=context, reason=reason,
                      track_id=self._approach_track_id, xy=self.approach_goal.tolist())
        self.approach_goal = None
        self._approach_track_id = None

    def withdraw(self, reason, context=None):
        if self.is_committed:
            # A cleared target map must earn a fresh commit, even for the same geometry.
            self._evidence = PositiveEvidenceStore(self.same_instance_m)
        super().withdraw(reason, context)

    def _check(self, stage, indices, candidates, rgb, context):
        if not indices:
            return []
        self.verification_calls += 1
        call_context = {**context, 'stage': stage, 'box_indices': indices}
        answers = self._verify(rgb, [candidates[i]['box'] for i in indices], call_context)
        if len(answers) != len(indices) or any(a not in ('yes', 'no', 'unknown') for a in answers):
            raise ValueError('Expected one valid verifier answer per box')
        passed = [i for i, answer in zip(indices, answers) if answer == 'yes']
        self._log('vlfm_' + stage, context=call_context, answers=answers, yes_box_indices=passed)
        return passed

    def _associate(self, candidates, kept, passed, failed_near, step):
        # A separately boxed failed neighbour can explain an old position better
        # than today's yes. Reserve that track for this frame: do NOT associate the
        # failed neighbour, move the track, erase its vote, or make a lasting veto.
        # Use the store's XY merge window and same-frame box overlap convention.
        reserved = []
        for track in self._evidence.tracks if failed_near else ():
            if not track.positive_steps:
                continue
            nearby = {i: np.linalg.norm(c['gps'] - track.gps) for i, c in candidates.items()
                      if np.max(np.abs(c['gps'] - track.gps)) <= self.same_instance_m}
            rivals = [i for i in kept if i in nearby and i in passed] or [i for i in kept if i in nearby]
            if rivals and any(j in nearby and all(nearby[j] <= nearby[i] and
                              _iou(candidates[i]['box'], candidates[j]['box']) < .5 for i in rivals)
                              for j in failed_near):
                reserved.append(track)

        # The store associates one box per track per frame. An exact detector repeat
        # is one observation, not two competing tracks. Keep its original box indices
        # so a qualifying commit still returns all the corresponding yes boxes.
        keys, unique = {}, {}
        for i in kept:
            candidate = candidates[i]
            candidate['lock_passed'] = i in passed
            key = tuple(candidate['box']), tuple(candidate['gps'])
            keys[i] = key
            if key not in unique or i in passed:
                unique[key] = candidate
        for track in reserved:
            track.vetoed = True
        try:
            tracks = self._evidence.associate(list(unique.values()), step)
        finally:
            for track in reserved:
                track.vetoed = False
        by_key = dict(zip(unique, tracks))
        return {i: by_key[key] for i, key in keys.items()}

    def select(self, proposals, rgb, context):
        if self.is_committed:
            return super().select(proposals, rgb, context)

        step = context['step']
        candidates = {}
        for i, p in enumerate(proposals):
            if (p.cloud is None or not len(p.cloud) or p.distance is None or
                    not np.isfinite(p.distance) or p.distance < 0):
                continue
            # One stable XY in the episodic map for tracking and temporary PointNav.
            xy = np.median(p.cloud[:, :2], axis=0)
            if np.isfinite(xy).all():
                candidates[i] = {'gps': xy, 'node_id': None, 'box': list(map(float, p.box))}

        admitted = list(candidates)
        if self.admission:
            admitted = self._check('candidate_admission', admitted, candidates, rgb, context)
        near = [i for i in admitted if proposals[i].distance < self.distance_threshold]
        yes = self._check('goal_confirmation', near, candidates, rgb, context)
        passed = set(yes)
        kept = [i for i in admitted if i in passed or proposals[i].distance >= self.distance_threshold]
        if not kept:
            return []  # A lost/denied frame does not cancel an earlier approach or vote.

        tracks = self._associate(candidates, kept, passed, set(near) - passed, step)
        qualified = []
        for i in yes:
            track = tracks[i]
            if step not in track.positive_steps:
                track.positive_steps.add(step)
                self._log('vlfm_candidate_vote', context=context, track_id=track.track_id,
                          box_index=i, count=len(track.positive_steps), threshold=self.threshold)
            if len(track.positive_steps) >= self.threshold:
                qualified.append(i)

        if qualified:
            self.release_approach('committed', context)
            self.committed = [proposals[i].cloud for i in qualified]
            self.commit_count += 1
            self._log('vlfm_commit', context=context, box_indices=qualified,
                      track_ids=[tracks[i].track_id for i in qualified], threshold=self.threshold,
                      distances_m=[round(float(proposals[i].distance), 3) for i in qualified])
            return qualified

        nearest = min(kept, key=lambda i: proposals[i].distance)
        track = tracks[nearest]
        self.approach_goal = track.gps.copy()
        if track.track_id != self._approach_track_id:
            event = 'vlfm_approach_start' if self._approach_track_id is None else 'vlfm_approach_switch'
            self._log(event, context=context, track_id=track.track_id, previous_track_id=self._approach_track_id,
                      xy=self.approach_goal.tolist(), box_index=nearest, distance_m=float(proposals[nearest].distance))
        self._approach_track_id = track.track_id
        return []
