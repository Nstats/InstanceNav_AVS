"""CoIN-Bench measures, ported from Context-Nav (vlfm/measurements/distractor_success.py).

DistractorSuccess is 1 on the step the agent calls STOP within
SUCCESS_DISTANCE + 0.1 m (geodesic) of any same-category distractor position,
else 0. Context-Nav reads the threshold from measurements.success.success_distance;
this fork's yacs config keeps it at TASK.SUCCESS.SUCCESS_DISTANCE. STOP is read
from task.is_stop_called, the flag Success uses, which is set exactly when the
STOP action (index 0, Context-Nav's `action != 0` test) is taken.
"""
from typing import Any

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry

# Context-Nav: "some objects (pictures, mirrors) are flush against walls".
DISTRACTOR_OFFSET_M = 0.1


@registry.register_measure
class DistractorSuccess(Measure):
    cls_uuid: str = 'distractor_success'

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._episode = None
        self._success_distance = float(kwargs['task']._config.SUCCESS.SUCCESS_DISTANCE)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        if len(episode.goals) != 1:
            raise ValueError('DistractorSuccess needs single-instance episodes')
        self._episode = episode
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = 0
        if not getattr(task, 'is_stop_called', False):
            return
        position = self._sim.get_agent_state().position
        for distractor in getattr(episode.goals[0], 'distractors', None) or []:
            # No episode argument: the episode's cached multi-goal path is for the
            # target view points and must not be reused here.
            distance = self._sim.geodesic_distance(position, distractor)
            if np.isnan(distance) or np.isinf(distance):
                continue
            if distance < self._success_distance + DISTRACTOR_OFFSET_M:
                self._metric = 1
                return
