"""Navigation adapter; exploration stays on the existing S6 candidate-retirement path."""
import math
from pathlib import Path

import numpy as np

from configs.navigation_config import relative_evidence_settings
from utils.log_utils import log_event
from utils.relative_evidence import CandidatePool, decision_snapshot, finite_position
from utils.relative_comparison import compare_candidates


class RelativeEvidenceNavigation:
    def __init__(self, agent, calibration, scene):
        self.agent = agent
        self.calibration = calibration
        self.model, self.model_id = calibration.for_scene(scene)
        self.settings = relative_evidence_settings()
        self.pool = CandidatePool()
        self.state = 'exploring'
        self.selected = None
        self.target = None
        self.recovery_origin = None
        self.last_snapshot = None
        self.log('Start', model_id=self.model_id, params=self.model['params'],
                 calibration_sha256=calibration.sha256,
                 source_exploration_description=calibration.data['source']['exploration_description'])

    def log(self, event, **fields):
        log_event('Relative ' + event, episode=getattr(self.agent, 'count_episodes', None),
                  policy_step=self.agent.total_steps, state=self.state, **fields)

    def observe(self, observations, boxes, candidates, decisions, source):
        if self.state != 'exploring':
            return
        for index, (box, candidate) in enumerate(zip(boxes, candidates)):
            # Stable identity of one detector observation, independent of API retry IDs.
            event_id = (self.agent.total_steps, source, index, tuple(float(x) for x in box))
            before = len(self.pool.seen_events)
            c = self.pool.add(event_id=event_id, step=self.agent.total_steps, gps=candidate['gps'],
                              answer=decisions.get(index), agent_position=observations['gps'],
                              observed_distance=candidate['distance'], rgb=observations['rgb'], box=box)
            if c is not None:
                self.log('Candidate', source=source, box_index=index, answer=decisions.get(index),
                         candidate=c.summary())
            elif len(self.pool.seen_events) != before:
                self.log('Invalid Localization', source=source, box_index=index)

    def on_reset(self, reason):
        if self.state == 'returning' and self.recovery_origin is None:
            position = finite_position(self.agent.last_gps)
            self.recovery_origin = position.copy() if position is not None else np.zeros(2)
        self.log('Navigation Reset', reason=reason, retained_candidates=len(self.pool.candidates))

    def restore_target(self):
        if self.state != 'returning':
            return
        if self.agent.in_stuck_recovery:
            return
        if self.recovery_origin is not None:
            if math.dist(self.agent.last_gps, self.recovery_origin) < .05:
                return  # Let existing turning/random escape finish before resuming the return.
            self.recovery_origin = None
            self.log('Return Resumed', candidate_id=self.selected)
        self.agent.found_goal = True
        self.agent.found_possible_goal = False
        self.agent.goal_gps = np.array(self.target, dtype=float)
        self.agent.landmark_navigation_steps = 0
        self.agent.detection_manager._emit_goal_lock_once('relative_evidence', candidate_id=self.selected)

    def tick(self):
        if self.state == 'finished':
            return True
        if self.state != 'exploring':
            return False
        env = self.agent.simulator._env
        elapsed = int(env._elapsed_steps)
        remaining = max(0, int(env._max_episode_steps) - elapsed)
        snapshot = decision_snapshot(self.pool, self.model, step=self.agent.total_steps,
                                     position=self.agent.last_gps, remaining=remaining,
                                     use_u=self.settings['use_u'], early_stop=self.settings['early_stop'],
                                     drop_rejected=self.settings['drop_rejected'])
        self.last_snapshot = snapshot
        self.log('Decision', atomic_steps=elapsed, **snapshot)
        if snapshot['trigger'] is None:
            return False
        candidates = [self.pool.candidates[i] for i in snapshot['shortlist']]
        if not candidates:
            self.finish('not_found')
            return True
        self.state = 'comparing'
        if self.settings['pairwise'] and len(candidates) > 1:
            winner, stats = compare_candidates(
                self.agent.scenegraph.model_client, candidates, self.agent.obj_goal, self.agent.goal_attributes,
                context={'episode': self.agent.count_episodes, 'policy_step': self.agent.total_steps})
        else:
            winner, stats = candidates[0], {'logical_requests': 0, 'format_retries': 0}
        self.selected, self.target = winner.candidate_id, winner.position
        self.state = 'returning'
        self.log('Selected', candidate_id=self.selected, gps=self.target,
                 shortlist=[c.summary() for c in candidates], **stats)
        # Persist the comparison evidence without embedding image bytes in logs.
        output = getattr(self.agent, 'visualization_dir', None)
        if output:
            directory = Path(output) / 'relative_evidence' / f'episode_{self.agent.count_episodes}'
            directory.mkdir(parents=True, exist_ok=True)
            for c in candidates:
                if c.best_view:
                    (directory / f'candidate_{c.candidate_id}.png').write_bytes(c.best_view['png'])
        self.restore_target()
        return False

    def finish(self, reason):
        if self.state == 'finished':
            return
        self.state = 'finished'
        self.log('Finished', reason=reason, candidate_id=self.selected,
                 atomic_steps=int(self.agent.simulator._env._elapsed_steps))

    def action(self, number_action):
        if number_action == 0:
            if self.state == 'returning' and self.agent.found_goal and self.recovery_origin is None:
                self.finish('planner_stop')
            elif self.state != 'finished':
                # An exploratory waypoint/escape STOP must not terminate the method early.
                self.log('Waypoint Stop Suppressed')
                return 3
        return number_action
