"""AVS instance decision for a base that keeps a target-object map (VLFM).

The base proposes detections of the goal category; AVS decides which may enter the
target map, i.e. which may become the navigation goal and end the episode with STOP:

* A detection is verified only within ``distance_threshold`` metres (4 m, as PSG-Nav).
  All eligible boxes of one frame go to the VLM in one request (numbered red boxes on
  the whole RGB frame, full description), through utils.instance_verifier.
* One yes commits: the yes boxes enter the target map and verification stops.
* no / unknown leave no trace: nothing is stored, the box is verified again when it is
  seen again within the distance. Farther boxes are ignored (A0: an unconfirmed
  candidate never becomes the goal).
* After the commit, detections of the same instance (within ``same_instance_m`` of the
  committed points) still refine the target map, as the base does for its target.
  Other detections are ignored. If the base later removes every committed point (VLFM
  clears ephemeral far detections), the commit is withdrawn and verification resumes.

This module has no VLFM or habitat imports, so it is unit-tested on its own.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class Proposal:
    box: Sequence[float]          # pixel xyxy on the RGB frame
    cloud: Optional[np.ndarray]   # global points (N, >=3) or None if unusable
    distance: Optional[float]     # metres from the camera to the closest point


class AVSDecision:
    def __init__(self, verify: Callable, distance_threshold: float = 4.0, same_instance_m: float = 0.8,
                 log: Optional[Callable] = None):
        self._verify = verify
        self.distance_threshold = float(distance_threshold)
        self.same_instance_m = float(same_instance_m)
        self._log = log or (lambda *args, **kwargs: None)
        self.reset()

    def reset(self):
        self.committed: List[np.ndarray] = []
        self.commit_count = 0
        self.verification_calls = 0

    @property
    def is_committed(self):
        return bool(self.committed)

    def withdraw(self, reason, context=None):
        if self.committed:
            self._log('vlfm_commit_withdrawn', context=context, reason=reason)
        self.committed = []

    def _same_instance(self, cloud):
        from scipy.spatial import cKDTree
        tree = cKDTree(np.concatenate([c[:, :2] for c in self.committed], axis=0))
        distances, _ = tree.query(cloud[:, :2], k=1, distance_upper_bound=self.same_instance_m)
        return bool(np.isfinite(distances).any())

    def select(self, proposals: Sequence[Proposal], rgb, context=None) -> List[int]:
        """Indices of the proposals that may enter the target map in this frame."""
        usable = [i for i, p in enumerate(proposals) if p.cloud is not None and len(p.cloud)]
        if self.committed:
            same = [i for i in usable if self._same_instance(proposals[i].cloud)]
            self.committed.extend(proposals[i].cloud for i in same)
            return same
        eligible = [i for i in usable if proposals[i].distance < self.distance_threshold]
        if not eligible:
            return []
        self.verification_calls += 1
        matches = self._verify(rgb, [list(map(float, proposals[i].box)) for i in eligible],
                               {**(context or {}), 'stage': 'goal_confirmation', 'box_indices': eligible})
        yes = [i for i, answer in zip(eligible, matches) if answer == 'yes']
        if yes:
            self.committed = [proposals[i].cloud for i in yes]
            self.commit_count += 1
            self._log('vlfm_commit', context=context, box_indices=yes,
                      distances_m=[round(float(proposals[i].distance), 3) for i in yes])
        return yes
