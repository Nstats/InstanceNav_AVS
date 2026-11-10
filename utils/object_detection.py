"""
Object Detection Module for PSG-Nav Agent
Handles object detection, goal detection, and RAG verification.
"""

import copy
import logging
from pathlib import Path
from uuid import uuid4
import torch
import numpy as np

from configs.detector_config import DETECTOR_TYPE
from configs.dataset_registry import is_instance_dataset
from configs.navigation_config import (
    instance_confirmation_enabled,
    instance_confirmation_threshold,
    verify_before_candidate_enabled,
    instance_confirmation_mode,
    post_confirmation_contradiction_check_enabled,
    preserve_positive_evidence_on_recovery_enabled,
    instance_collect_only_enabled,
    relative_evidence_enabled,
    validate_relative_evidence,
    instance_rejection_policy,
    validate_instance_ablation,
    instance_confirmation_count_unit,
)
from utils import collect_only
from utils.log_utils import log_event, vprint

LOG = logging.getLogger(__name__)


class ObjectDetectionManager:
    """
    Manages object detection and goal finding for the PSG-Nav agent.
    This class encapsulates all detection-related logic to improve code readability.
    """

    def __init__(self, agent):
        """
        Initialize the ObjectDetectionManager.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        self.agent = agent

    def _emit_goal_lock_once(self, path, **fields):
        """Emit one [Goal Lock] record per lock period (rising edge only)."""
        if getattr(self.agent, '_goal_lock_active', False):
            return
        self.agent._goal_lock_active = True
        gps = np.asarray(getattr(self.agent, 'goal_gps', [0., 0.]), dtype=float)
        try:
            threshold = self.agent.get_goal_detection_threshold()
        except Exception:
            threshold = None
        log_event('Goal Lock', episode=getattr(self.agent, 'count_episodes', None),
                  step=getattr(self.agent, 'total_steps', None), path=path,
                  goal_gps=gps.tolist(),
                  found_goal_times=int(getattr(self.agent, 'found_goal_times', 0)),
                  threshold=threshold, **fields)

    def _emit_detection(self, source, *, kept, box=None, category=None, distance=None,
                        gps=None, center=None, verification_id=None, prediction_idx=None):
        """Structured per-candidate record; enables offline spatial candidate tracking."""
        est_map = None
        if gps is not None:
            try:
                gy, gx = float(gps[0]), float(gps[1])
                if np.isfinite(gy) and np.isfinite(gx):
                    est_map = [int(self.agent.map_size_cm / 10 + gy * 100 / self.agent.resolution),
                               int(self.agent.map_size_cm / 10 - gx * 100 / self.agent.resolution)]
            except (TypeError, IndexError):
                est_map = None
        log_event('Detection', episode=getattr(self.agent, 'count_episodes', None),
                  step=getattr(self.agent, 'total_steps', None),
                  source=source, category=category or getattr(self.agent, 'obj_goal', None),
                  kept=bool(kept), box=[int(v) for v in box] if box is not None else None,
                  est_distance_m=(round(float(distance), 3)
                                  if distance is not None and np.isfinite(float(distance)) else None),
                  est_gps=([round(float(gps[0]), 3), round(float(gps[1]), 3)]
                           if gps is not None
                           and np.all(np.isfinite([float(gps[0]), float(gps[1])])) else None),
                  est_map=est_map,
                  center_px=[int(center[0]), int(center[1])] if center is not None else None,
                  verification_id=verification_id,
                  prediction_idx=prediction_idx)

    def reset_instance_verification(self, episode):
        """Episode reset only: navigation recovery must retain semantic vetoes."""
        validate_instance_ablation()
        self.agent.goal_constraints = []
        self.agent.instance_rejections = []
        self._positive_evidence = None
        self._post_frame_tracks = []
        self._post_goal_track_id = None
        self._post_possible_track_id = None
        from utils.instance_evidence import RecoveryEvidenceGuard
        self._recovery_guard = RecoveryEvidenceGuard()
        self._recovery_decisions = {}
        # Collect-only mode (S6): tracks whose intended lock was logged; never approached again.
        self._collect_retired = set()
        self._collect_retired_gps = []
        self._collect_snapshot_step = None
        self._collection_recorder = None
        if self.collect_only_enabled():
            from utils.collection_recorder import CollectionRecorder
            self._collection_recorder = CollectionRecorder(self.agent, episode)
        if self._evidence_tracking_enabled():
            from utils.instance_evidence import PositiveEvidenceStore
            self._positive_evidence = PositiveEvidenceStore(self.agent.goal_merge_threshold)
        self.agent.relative_evidence_controller = None
        if relative_evidence_enabled():
            from utils.relative_evidence import Calibration
            from utils.relative_evidence_navigation import RelativeEvidenceNavigation
            settings = validate_relative_evidence(dataset=self.agent.dataset,
                                                 rag_mode=getattr(self.agent.args, 'rag_mode', 0),
                                                 backend=self.agent.scenegraph.model_client.backend)
            if (not hasattr(self, '_relative_calibration')
                    or self._relative_calibration.path != str(Path(settings['calibration']).resolve())):
                self._relative_calibration = Calibration(settings['calibration'])
            self.agent.relative_evidence_controller = RelativeEvidenceNavigation(
                self.agent, self._relative_calibration, episode.scene_id)

    def _post_check_enabled(self):
        return (is_instance_dataset(self.agent.dataset)
                and post_confirmation_contradiction_check_enabled())

    def preserve_evidence_on_recovery_enabled(self):
        return (is_instance_dataset(self.agent.dataset)
                and preserve_positive_evidence_on_recovery_enabled())

    def _evidence_tracking_enabled(self):
        return is_instance_dataset(self.agent.dataset) and instance_confirmation_enabled()

    def collect_only_enabled(self):
        return is_instance_dataset(self.agent.dataset) and instance_collect_only_enabled()

    def defer_goal_lock(self):
        return self.collect_only_enabled() or getattr(self.agent, 'relative_evidence_controller', None) is not None

    def _collect_retired_ids(self):
        return getattr(self, '_collect_retired', set())

    def _is_collect_retired(self, track):
        return track.track_id in self._collect_retired_ids()

    def _near_collect_retired(self, track):
        """A new track at a retired lock position (depth noise split the track) is not approached again."""
        if track.gps is None:
            return False
        return any(self._same_instance_position(track.gps, gps)
                   for gps in getattr(self, '_collect_retired_gps', ()))

    def _collect_retire(self, track, path):
        """Collect-only replacement for a lock: log it, retire the track, keep exploring."""
        retired = self._collect_retired_ids()
        controller = getattr(self.agent, 'relative_evidence_controller', None)
        if controller is None:
            collect_only.log_intended_lock(self.agent, path=path, track=track,
                                           store=self._positive_evidence, retired_ids=retired)
            if self._collection_recorder is not None:
                self._collection_recorder.intended_lock(track, path)
        else:
            controller.log('Deferred Lock', path=path, navigation_track_id=track.track_id,
                           gps=track.gps.tolist())
        retired.add(track.track_id)
        self._collect_retired = retired
        self._collect_retired_gps = [*getattr(self, '_collect_retired_gps', []), track.gps.copy()]
        if controller is None:
            collect_only.snapshot(self.agent, reason='intended_lock', store=self._positive_evidence,
                                  retired_ids=retired)
        self.agent.explanation = (f'Collect-only: logged intended lock on candidate {track.track_id}, exploring on'
                                  if controller is None else f'Deferred lock on candidate {track.track_id}, exploring on')

    def _collect_periodic_snapshot(self):
        step = getattr(self.agent, 'total_steps', None)
        if (step is None or step % collect_only.SNAPSHOT_EVERY != 0
                or getattr(self, '_collect_snapshot_step', None) == step):
            return
        self._collect_snapshot_step = step
        collect_only.snapshot(self.agent, reason='periodic', store=getattr(self, '_positive_evidence', None),
                              retired_ids=self._collect_retired_ids())

    def _agent_position(self):
        position = np.asarray(getattr(self.agent, 'last_gps', None), dtype=float)
        return position.copy() if position.shape == (2,) and np.all(np.isfinite(position)) else None

    def _recovery_log(self, event, **fields):
        from utils.instance_verifier import log_instance_verification
        log_instance_verification(event, context={
            'mode': 'recovery_preservation', 'episode': getattr(self.agent, 'count_episodes', None),
            'step': self.agent.total_steps,
        }, **fields)

    def reset_post_confirmation_progress(self, *, reason='unspecified', preserve=False):
        if getattr(self, '_positive_evidence', None) is not None:
            counts = {str(track.track_id): len(track.votes) for track in self._positive_evidence.tracks}
            active_id = self._post_goal_track_id
            if active_id is None:
                active_id = self._post_possible_track_id
            if preserve:
                self._recovery_guard.start(active_id, self._agent_position(), self.agent.total_steps)
                self._recovery_decisions.clear()
            else:
                self._positive_evidence.reset_progress()
            if self.preserve_evidence_on_recovery_enabled():
                self._recovery_log('evidence_reset', reason=reason, preserved=preserve,
                                   active_track_id=active_id, candidate_counts_before=counts)
        self._post_frame_tracks = []
        self._post_goal_track_id = None
        self._post_possible_track_id = None

    def _candidate_can_be_approached(self, track):
        if not self.preserve_evidence_on_recovery_enabled():
            return True
        reason = self._recovery_guard.blocked_reason(
            track.track_id, self._agent_position(), self.agent.total_steps,
            recovering=getattr(self.agent, 'in_stuck_recovery', False),
            stuck=(getattr(self.agent, 'forward_no_move_count', 0) >= 3
                   or getattr(self.agent, 'not_move_steps', 0) >= 12))
        if self._recovery_guard.step >= 0 and self._recovery_decisions.get(track.track_id, 'unseen') != reason:
            self._recovery_log('recovery_eligibility', track_id=track.track_id,
                               reason=reason or 'fresh_match_after_progress',
                               eligible=reason is None, retained_count=len(track.votes))
            self._recovery_decisions[track.track_id] = reason
        return reason is None

    def detect_objects(self, observations):
        """
        Main object detection function that runs the configured detector (GLIP or FastSAM)
        and checks for goal objects.

        Args:
            observations: Current observations dict containing RGB, depth, GPS, etc.
        """
        # Verification linkage is per-frame; reset before any new verification runs.
        self._last_verification_id = None
        # Run object detection using configured detector
        if DETECTOR_TYPE == 'glip':
            self._detect_with_glip(observations)
        elif DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
            self._detect_with_fastsam(observations)
        else:
            raise ValueError(f"Unknown detector type: {DETECTOR_TYPE}")

        # Process detections for all object categories
        self._process_all_object_detections(observations)

        if self.collect_only_enabled():
            self._collect_periodic_snapshot()

        # Handle goal detection based on object type
        if self.agent.scenegraph.obj_goal in self.agent.scenegraph.small_objects:
            # Small objects: use GroundingDINO scene graph detection
            self._detect_goal_with_scenegraph(observations)
        else:
            # Large objects: use GLIP/FastSAM navigation detector
            self._detect_goal_with_navigation_detector(observations)

    def _detect_with_glip(self, observations):
        """Run GLIP detector on current observation."""
        from utils.utils_glip import categories_21
        import cv2
        import os

        # Save original image before detection
        image_bgr = observations["rgb"][:,:,[2,1,0]].copy()
        self._save_detection_images(image_bgr, observations, before_detection=True)

        self.agent.current_obj_predictions = self.agent.glip_demo.inference(
            image_bgr,
            categories_21
        )
        new_labels = self._get_glip_real_label(self.agent.current_obj_predictions)
        self.agent.current_obj_predictions.add_field("labels", new_labels)

        # Print detection details
        num_detections = len(self.agent.current_obj_predictions)

        if num_detections > 0:
            scores = self.agent.current_obj_predictions.get_field('scores').cpu().numpy()
            bboxes = self.agent.current_obj_predictions.bbox.cpu().numpy()


            # Save image with bboxes after detection
            self._save_detection_images(image_bgr, observations, before_detection=False,
                                       bboxes=bboxes, labels=new_labels, scores=scores)

    def _detect_with_fastsam(self, observations):
        """Run FastSAM detector on current observation."""
        from configs.detector_config import CATEGORIES_21
        from utils.utils_glip import categories_21_origin

        image_rgb = observations["rgb"][:,:,[2,1,0]]  # Convert BGR to RGB
        results = self.agent.detector.detect(image_rgb, categories_21_origin)

        # Convert FastSAM results to GLIP-like format for compatibility
        class MockPredictions:
            def __init__(self, bboxes, labels, scores):
                self.bbox = torch.tensor(bboxes) if not isinstance(bboxes, torch.Tensor) else bboxes
                # Convert labels to tensor if it's a list (FastSAM returns list of strings)
                if isinstance(labels, list):
                    # Create a wrapper that has .tolist() method
                    class LabelWrapper:
                        def __init__(self, label_list):
                            self.labels = label_list
                        def tolist(self):
                            return self.labels
                        def __len__(self):
                            return len(self.labels)
                        def __getitem__(self, idx):
                            return self.labels[idx]
                    labels = LabelWrapper(labels)
                self._fields = {'labels': labels, 'scores': torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores}

            def get_field(self, name):
                return self._fields.get(name)

            def add_field(self, name, value):
                self._fields[name] = value

        self.agent.current_obj_predictions = MockPredictions(
            results['bboxes'],
            results['labels'],
            results['scores']
        )

        obj_labels = results['labels']
        print(f"[FastSAM Detection] Found {len(obj_labels)} objects")
        print(f"  bbox shape: {self.agent.current_obj_predictions.bbox.shape}")
        print(f"  scores shape: {self.agent.current_obj_predictions.get_field('scores').shape}")
        print(f"  labels: {obj_labels[:5]}...")  # First 5 labels

    def _process_all_object_detections(self, observations):
        """
        Process detections for all object categories and update obj_locations.

        Args:
            observations: Current observations
        """
        from utils.utils_glip import categories_21_origin

        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        category_to_index = {
            category_name: idx for idx, category_name in enumerate(categories_21_origin)
        }

        for j, label in enumerate(obj_labels):
            if label in category_to_index:
                confidence = self.agent.current_obj_predictions.get_field("scores")[j]
                bbox = self.agent.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = self._pixel_direction(center_point[0])
                temp_distance = self.agent.depth[center_point[1],center_point[0],0]


                if temp_distance >= self.agent.distance_threshold:
                    continue

                obj_gps = self.agent.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.agent.map_size_cm/10 - obj_gps[1]*100/self.agent.resolution)
                y = int(self.agent.map_size_cm/10 + obj_gps[0]*100/self.agent.resolution)
                self.agent.obj_locations[category_to_index[label]].append([confidence, x, y])

    def _detect_goal_with_scenegraph(self, observations):
        """
        Detect goal using GroundingDINO scene graph (for small objects).

        Args:
            observations: Current observations
        """
        # Using GroundingDINO scene graph detection for small objects
        self.agent.active_detection_path = 'scenegraph'

        self.agent.segment_num = len(self.agent.scenegraph.segment2d_results)
        goal_mask = []
        post_tracks = None

        if self.agent.segment_num > self.agent.last_segment_num:
            self.agent.last_segment_num = self.agent.segment_num
            segment2d_result = self.agent.scenegraph.segment2d_results[-1]
            indices = []
            goal_nodes = []

            for index, element in enumerate(segment2d_result['caption']):
                if self.agent.obj_goal_sg in element.split(' '):
                    for node in self.agent.scenegraph.nodes:
                        if node.is_goal_node and node.object['image_idx'][-1] == len(self.agent.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                            indices.append(index)
                            goal_nodes.append(node)

            goal_mask = [segment2d_result['mask'][index] for index in indices]
            if is_instance_dataset(self.agent.dataset) and goal_mask:
                boxes = []
                for mask in goal_mask:
                    ys, xs = np.where(mask)
                    boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                kept = self._verify_instance_boxes(observations, boxes, masks=goal_mask, nodes=goal_nodes)
                goal_mask = [goal_mask[i] for i in kept]
                if self._evidence_tracking_enabled():
                    post_tracks = [self._post_frame_tracks[i] for i in kept]


        if len(goal_mask) > 0:
            if post_tracks is not None:
                self._process_scenegraph_post_goals(post_tracks, observations, segment2d_result)
            else:
                self._process_scenegraph_goal_masks(goal_mask, observations, segment2d_result)
        else:
            # No goal detected, reset if previously found
            if self.agent.found_goal:
                self.agent.reset_goal_detection(reason='no_goal_observation')

    def _process_scenegraph_goal_masks(self, goal_mask, observations, segment2d_result):
        """
        Process goal masks from scene graph detection.

        Args:
            goal_mask: List of goal masks
            observations: Current observations
            segment2d_result: Scene graph segment result
        """
        possible_goal_detected_before = copy.deepcopy(self.agent.found_possible_goal)
        shortest_distance = 120
        shortest_distance_angle = 0

        for mask in goal_mask:
            center_point, temp_direction, temp_distance = self._scenegraph_goal_geometry(mask)
            if temp_distance >= self.agent.distance_threshold:
                self.agent.found_possible_goal = True
                self.agent.explanation = f"Potential goal detected: {self.agent.obj_goal} at {temp_distance:.2f}m away (beyond {self.agent.distance_threshold}m threshold)"
                # Update angle for GPS calculation even when goal is far
                if temp_distance < shortest_distance:
                    shortest_distance = temp_distance
                    shortest_distance_angle = temp_direction
            else:
                # Goal found within threshold
                self._handle_scenegraph_goal_found(
                    center_point, temp_direction, temp_distance,
                    observations, segment2d_result
                )

                # Update shortest distance for GPS calculation
                if temp_distance < shortest_distance:
                    shortest_distance = temp_distance
                    shortest_distance_angle = temp_direction

        # Update goal GPS if found
        if self.agent.found_goal:
            self.agent.goal_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            self.agent.explanation = f"Goal found: {self.agent.obj_goal} at {shortest_distance:.2f}m away (detection #{self.agent.found_goal_times})"
            self._emit_goal_lock_once('scenegraph')
        elif self.agent.found_possible_goal:
            # Update possible goal GPS whenever we have a valid possible goal
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            print(f"[SceneGraph GPS Update] Updated possible_goal_temp_gps with angle={shortest_distance_angle:.2f}°, distance={shortest_distance:.2f}m -> gps={self.agent.possible_goal_temp_gps}")

    def _handle_scenegraph_goal_found(self, center_point, temp_direction, temp_distance, observations, segment2d_result):
        """
        Handle goal found from scene graph detection (including RAG verification).

        Args:
            center_point: Center point of goal mask
            temp_direction: Direction to goal
            temp_distance: Distance to goal
            observations: Current observations
            segment2d_result: Scene graph segment result
        """
        # Calculate goal GPS and map coordinates for false positive marking
        goal_gps = self.agent.get_goal_gps(observations, temp_direction, temp_distance)
        map_x = int(self.agent.map_size_cm/10 - goal_gps[1]*100/self.agent.resolution)
        map_y = int(self.agent.map_size_cm/10 + goal_gps[0]*100/self.agent.resolution)

        self._emit_detection('scenegraph', kept=True, category=self.agent.obj_goal,
                             distance=temp_distance, gps=goal_gps, center=center_point,
                             verification_id=getattr(self, '_last_verification_id', None))

        # Check if this position is marked as false positive (value == -1)
        if 0 <= map_y < self.agent.map_size and 0 <= map_x < self.agent.map_size:
            if self.agent.goal_gps_map[map_y, map_x] == -1:
                print(f"[False Positive Skip] SceneGraph position map({map_y},{map_x}) is marked as false positive, skipping detection")
                return  # Skip this detection entirely

        # Extract crop for RAG
        bbox_int = [
            int(center_point[0] - 50), int(center_point[1] - 50),
            int(center_point[0] + 50), int(center_point[1] + 50)
        ]
        bbox_int[0] = max(0, bbox_int[0])
        bbox_int[1] = max(0, bbox_int[1])
        bbox_int[2] = min(observations["rgb"].shape[1], bbox_int[2])
        bbox_int[3] = min(observations["rgb"].shape[0], bbox_int[3])
        crop = observations["rgb"][bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]

        # Get detected caption for debugging
        detected_caption = segment2d_result['caption'][0] if 'caption' in segment2d_result else self.agent.obj_goal_sg
        initial_conf = 0.9

        vprint('rag', f"[RAG Window] Adding scenegraph detection: detected_caption='{detected_caption}', using caption='{self.agent.obj_goal}' (goal object)")

        # Update sliding window - CRITICAL: use obj_goal as caption
        detection_dict = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detected caption
            'confidence': initial_conf,
            'distance': temp_distance,
            'step': self.agent.total_steps
        }
        self.agent.rag_sliding_window.append(detection_dict)

        # Keep only latest N detections
        if len(self.agent.rag_sliding_window) > self.agent.rag_sliding_window_size:
            self.agent.rag_sliding_window.pop(0)

        # Update found_goal state
        if self.agent.found_goal:
            if temp_distance < self.agent.distance_threshold:
                self.agent.found_goal_times = self.agent.found_goal_times + 1

        self.agent.found_goal = True
        self.agent.found_possible_goal = False

        # Update latest goal detection - also use obj_goal and include map coordinates
        self.agent.rag_latest_goal_detection = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detected caption
            'confidence': initial_conf,
            'distance': temp_distance,
            'verified': True,
            'map_x': map_x,  # Store map coordinates for false positive marking
            'map_y': map_y
        }

        # RAG verification
        if self.agent.rag_check_enabled and self.agent.rag_latest_goal_detection is not None:
            self._verify_detection_with_rag()

    def _detect_goal_with_navigation_detector(self, observations):
        """
        Detect goal using GLIP/FastSAM navigation detector (for large objects).

        Args:
            observations: Current observations
        """
        # Using GLIP/FastSAM navigation detector for large objects
        self.agent.active_detection_path = 'navigation'

        # Find goal bboxes
        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        goal_bbox = []
        goal_bbox_indices = []
        post_tracks = None

        for j, label in enumerate(obj_labels):
            # Exact match or word-level match (avoid substring false positives)
            label_lower = label.lower() if isinstance(label, str) else str(label).lower()
            goal_lower = self.agent.obj_goal.lower()

            # Check if goal matches (exact match or as separate word)
            from configs.dataset_categories import goal_label_matches
            is_match = goal_label_matches(label_lower, goal_lower, self.agent.dataset)

            if is_match:
                vprint('detection', f"[Goal Match] Matched '{label}' as goal '{self.agent.obj_goal}'")
                goal_bbox.append(self.agent.current_obj_predictions.bbox[j])
                goal_bbox_indices.append(j)
            elif self.agent.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.agent.current_obj_predictions.bbox[j])
                goal_bbox_indices.append(j)

        if DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
            print(f"[Goal Matching] obj_goal='{self.agent.obj_goal}', found {len(goal_bbox)} matches out of {len(obj_labels)} detections")
            if len(goal_bbox) > 0:
                matched_labels = [obj_labels[idx] if hasattr(obj_labels, '__getitem__') else obj_labels.tolist()[idx] for idx in goal_bbox_indices]
                print(f"  Matched labels: {matched_labels}")

        if is_instance_dataset(self.agent.dataset) and goal_bbox:
            kept = self._verify_instance_boxes(observations, [box.tolist() for box in goal_bbox],
                                               prediction_indices=goal_bbox_indices)
            goal_bbox = [goal_bbox[i] for i in kept]
            goal_bbox_indices = [goal_bbox_indices[i] for i in kept]
            if self._evidence_tracking_enabled():
                post_tracks = [self._post_frame_tracks[i] for i in kept]
            if not kept and self.agent.found_goal:
                self.agent.reset_goal_detection(reason='instance_rejected')

        if len(goal_bbox) > 0:
            self._process_navigation_goal_bboxes(goal_bbox, goal_bbox_indices, observations,
                                                post_tracks=post_tracks)

    def _instance_candidates(self, observations, boxes, *, masks, nodes, prediction_indices):
        candidates = []
        for index, box in enumerate(boxes):
            if masks is not None:
                center, direction, distance = self._scenegraph_goal_geometry(masks[index])
                gps = self.agent.get_goal_gps(observations, direction, distance)
            else:
                center, direction, distance, gps = self._navigation_goal_geometry(
                    torch.as_tensor(box), prediction_indices[index], observations)
            gps = np.asarray(gps, dtype=float)
            if (gps.shape != (2,) or not np.all(np.isfinite(gps))
                    or not np.isfinite(float(distance)) or not 0 < float(distance) < 100):
                gps = None
            candidates.append({
                'node_id': nodes[index].object.get('id', id(nodes[index])) if nodes is not None else None,
                'gps': gps, 'box': list(box), 'center': center,
                'direction': direction, 'distance': float(distance),
            })
        return candidates

    def _verify_instance_boxes(self, observations, boxes, *, masks=None, nodes=None, prediction_indices=None):
        if not instance_confirmation_enabled():
            return list(range(len(boxes)))
        from utils.instance_verifier import log_instance_verification, verify_candidates, verify_candidate_decisions
        controller = getattr(self.agent, 'relative_evidence_controller', None)
        decisions = {}
        conflict = instance_confirmation_mode() == 'contradiction'
        rejection_policy = instance_rejection_policy()
        candidates = self._instance_candidates(observations, boxes, masks=masks, nodes=nodes,
                                               prediction_indices=prediction_indices)
        context = {'episode': getattr(self.agent, 'count_episodes', None),
                   'verification_id': uuid4().hex,
                   'step': self.agent.total_steps, 'category': self.agent.obj_goal,
                   'mode': 'contradiction' if conflict else 'instance_match'}
        collection = self._collection_recorder if self.collect_only_enabled() else None
        stages = []
        if collection is not None:
            scores = []
            for i in range(len(boxes)):
                score = None
                if masks is None:
                    values = self.agent.current_obj_predictions.get_field('scores')
                    score = float(values[prediction_indices[i]])
                elif nodes is not None:
                    result = self.agent.scenegraph.segment2d_results[-1]
                    values = result.get('confidence')
                    if values is not None:
                        score = float(values[nodes[i].object['mask_idx'][-1]])
                scores.append(score if score is not None and np.isfinite(score) else None)
            context.update(collection.begin_frame(observations, candidates,
                           'scenegraph' if masks is not None else 'navigation', scores))
        vetoes = []
        veto_indices = set()
        veto_answers = {}

        def rejected(candidate):
            if self._instance_was_rejected(candidate):
                return True
            return any(self._candidate_matches(candidate, veto) for veto in vetoes)

        def verify_stage(indices, stage):
            if not indices:
                return []
            stage_context = {**context, 'verification_id': uuid4().hex,
                             'stage': stage, 'box_indices': indices}
            if controller is not None or collection is not None or rejection_policy != 'none':
                matches = verify_candidate_decisions(
                    self.agent.scenegraph.model_client, observations['rgb'], [boxes[i] for i in indices],
                    self.agent.obj_goal, self.agent.goal_attributes, log_context=stage_context)
                decisions.update(zip(indices, matches))
                if collection is not None:
                    stages.append({'stage': stage, 'verification_id': stage_context['verification_id'],
                                   'box_indices': list(indices), 'matches': matches})
                local_kept = {i for i, answer in enumerate(matches) if answer == 'yes'}
                if stage == 'goal_confirmation' and rejection_policy != 'none':
                    for i, answer in zip(indices, matches):
                        if answer == 'no' or (answer == 'unknown' and rejection_policy == 'non_yes'):
                            vetoes.append(candidates[i])
                            veto_indices.add(i)
                            veto_answers[i] = answer
            else:
                local_kept = set(verify_candidates(
                    self.agent.scenegraph.model_client, observations['rgb'], [boxes[i] for i in indices],
                    self.agent.obj_goal, self.agent.goal_attributes, conflict=conflict,
                    log_context=stage_context))
            if conflict:
                vetoes.extend(candidates[i] for j, i in enumerate(indices) if j not in local_kept)
                veto_indices.update(i for j, i in enumerate(indices) if j not in local_kept)
                veto_answers.update({i: 'yes' for j, i in enumerate(indices) if j not in local_kept})
            self._last_verification_id = stage_context['verification_id']
            return [i for j, i in enumerate(indices) if j in local_kept and not rejected(candidates[i])]

        active = [i for i, candidate in enumerate(candidates) if not rejected(candidate)]
        if len(active) != len(boxes):
            log_instance_verification('history_skip', context=context,
                                      skipped_box_indices=[i for i in range(len(boxes)) if i not in active],
                                      reason='previous_contradiction' if conflict else 'permanent_rejection',
                                      rejection_policy=rejection_policy)
        # Admission is independent: passing it never produces a confirmation vote.
        # With admission disabled, detector proposals may be approached directly.
        preverify = verify_before_candidate_enabled()
        admitted = verify_stage(active, 'candidate_admission') if preverify else active
        eligible = [i for i in admitted if candidates[i]['gps'] is not None
                    and candidates[i]['distance'] < self.agent.distance_threshold]
        confirmed = verify_stage(eligible, 'goal_confirmation')

        if controller is not None:
            controller.observe(observations, boxes, candidates, decisions,
                               'scenegraph' if masks is not None else 'navigation')

        # Commit semantic vetoes only after all requested checks succeed. A failed
        # API call/invalid response must not create votes or partial veto history.
        for index in sorted(veto_indices):
            candidate = candidates[index]
            self.agent.instance_rejections.append({**candidate, 'step': self.agent.total_steps,
                                                  'decision': veto_answers[index],
                                                  'policy': 'contradiction' if conflict else rejection_policy})
            self._clear_rejected_instance(candidate)
            if not conflict:
                log_instance_verification('permanent_rejection', context=context,
                                          box_index=index, answer=veto_answers[index],
                                          rejection_policy=rejection_policy,
                                          node_id=str(candidate['node_id']) if candidate['node_id'] is not None else None,
                                          gps=candidate['gps'].tolist() if candidate['gps'] is not None else None)
        for index in sorted(veto_indices):
            candidate = candidates[index]
            tag = 'Contradiction Reject' if conflict else 'Instance Reject'
            print(f'[{tag}] node={candidate["node_id"]} gps={candidate["gps"]} '
                  f'box={index}')
            if candidate['gps'] is None and candidate['node_id'] is None:
                LOG.warning('Rejected unlocalized box %d; history cannot associate it across views', index)
        confirmed = [i for i in confirmed if not rejected(candidates[i])]
        # Near candidates denied by the independent confirmation check are not
        # counted or associated. In particular, a rejected duplicate cannot steal
        # an existing track from the box that passed.
        kept = [i for i in admitted if not rejected(candidates[i])
                and (i not in eligible or i in confirmed)]
        positive = set(admitted if preverify else ()) | set(confirmed)
        self._record_candidate_observations(observations, boxes, candidates, kept,
                                            confirmed=set(confirmed), positive=positive)
        if self.collect_only_enabled():
            collect_only.log_frame(self.agent, candidates=candidates, tracks=self._post_frame_tracks,
                                   kept=kept, confirmed=set(confirmed), positive=positive,
                                   retired_ids=self._collect_retired_ids())
            collection.finish_frame(context, self._post_frame_tracks, kept, stages,
                                    self._collect_retired_ids())
        log_instance_verification('filter_result', context={**context, 'stage': 'candidate_pipeline'},
                                  checked_box_indices=active if preverify else eligible,
                                  contradicted_box_indices=sorted(veto_indices),
                                  admitted_box_indices=admitted, confirmation_box_indices=eligible,
                                  confirmed_box_indices=confirmed, kept_box_indices=kept,
                                  rejected_box_indices=[i for i in range(len(boxes)) if i not in kept])
        if rejection_policy != 'none':
            log_instance_verification('rejection_policy_frame', context=context,
                rejection_policy=rejection_policy, candidates=[{
                    'box_index': i, 'box': list(map(float, c['box'])),
                    'node_id': str(c['node_id']) if c['node_id'] is not None else None,
                    'gps': c['gps'].tolist() if c['gps'] is not None else None,
                    'answer': decisions.get(i), 'history_skipped': i not in active,
                    'permanently_rejected': rejected(c),
                } for i, c in enumerate(candidates)])
        self._emit_rejected_detections(boxes, kept, masks=masks,
                                      prediction_indices=prediction_indices,
                                      verification_id=getattr(self, '_last_verification_id', None),
                                      candidate_gps=[c['gps'] for c in candidates])
        return kept

    def _candidate_matches(self, candidate, other):
        if candidate['node_id'] is not None and other['node_id'] is not None:
            return candidate['node_id'] == other['node_id']
        return self._same_instance_position(candidate['gps'], other['gps'])

    def _emit_rejected_detections(self, boxes, kept, *, masks, prediction_indices,
                                  verification_id, candidate_gps=None):
        """[Detection] records for boxes filtered out by instance verification."""
        kept_set = set(kept)
        source = 'scenegraph' if masks is not None else 'navigation'
        for i in range(len(boxes)):
            if i in kept_set:
                continue
            gps = None
            if candidate_gps is not None and candidate_gps[i] is not None:
                gps = candidate_gps[i]
            self._emit_detection(source, kept=False, box=boxes[i],
                                 gps=gps,
                                 verification_id=verification_id,
                                 prediction_idx=(prediction_indices[i]
                                                 if prediction_indices is not None
                                                 and i < len(prediction_indices) else None))

    def _record_candidate_observations(self, observations, boxes, candidates, kept, *, confirmed, positive):
        # Associate passed confirmation boxes first. Unverified far candidates
        # may be approached, but must not claim a positive box's existing track.
        ordered = sorted(kept, key=lambda index: index not in confirmed)
        for index in kept:
            candidates[index]['lock_passed'] = index in confirmed
        tracks = self._positive_evidence.associate([candidates[i] for i in ordered], self.agent.total_steps)
        self._post_frame_tracks = [None] * len(boxes)
        for index, track in zip(ordered, tracks):
            self._post_frame_tracks[index] = track
            if index in positive:
                track.positive_steps.add(self.agent.total_steps)
                if self._post_check_enabled():
                    track.record_positive(observations['rgb'], self.agent.total_steps, boxes[index])
            if self.preserve_evidence_on_recovery_enabled():
                self._recovery_log('positive_association' if index in positive else 'candidate_association',
                                   track_id=track.track_id, box_index=index, retained_count=len(track.votes))

    def _post_candidate_is_available(self, track):
        if track.vetoed or track.gps is None or track.rejected_step == self.agent.total_steps:
            return False
        x = int(self.agent.map_size_cm / 10 - track.gps[1] * 100 / self.agent.resolution)
        y = int(self.agent.map_size_cm / 10 + track.gps[0] * 100 / self.agent.resolution)
        return (0 <= x < self.agent.map_size and 0 <= y < self.agent.map_size
                and self.agent.goal_gps_map[y, x] != -1)

    def _rebuild_post_goal_counts(self):
        # Undo only a vetoed candidate's contributions, preserving the original
        # spatial merge cells and repeated-detection counting for all other objects.
        positive = self.agent.goal_gps_map > 0
        self.agent.goal_gps_map[positive] = 0
        self.agent.goal_gps_timestamp_map[positive] = 0
        for track in self._positive_evidence.tracks:
            for y, x, step in track.votes:
                if self.agent.goal_gps_map[y, x] == -1:
                    continue
                self.agent.goal_gps_map[y, x] += 1
                self.agent.goal_gps_timestamp_map[y, x] = max(
                    self.agent.goal_gps_timestamp_map[y, x], step)
        self.agent.found_goal_times = self.agent.goal_gps_map.max()

    def _post_recheck(self, track):
        from utils.instance_verifier import log_instance_verification, verify_post_confirmation
        images, frames = track.images_and_metadata()
        if not frames or frames[-1]['step'] != self.agent.total_steps:
            raise RuntimeError('Final confirmation requires the current positive observation')
        key = tuple(frame['sha256'] for frame in frames)
        context = {'episode': getattr(self.agent, 'count_episodes', None),
                   'step': self.agent.total_steps, 'track_id': track.track_id,
                   'observation_steps': [frame['step'] for frame in frames],
                   'boxes': [frame['box'] for frame in frames]}
        if key == track.last_check_key:
            result = track.last_check_result
            log_instance_verification('recheck_reuse', context=context, **result)
        else:
            result = verify_post_confirmation(self.agent.scenegraph.model_client, images,
                                              self.agent.obj_goal, self.agent.goal_attributes,
                                              log_context=context)
            track.last_check_key, track.last_check_result = key, result
        if result['decision'] != 'conflict':
            return True
        track.votes.clear()
        track.rejected_step = self.agent.total_steps
        self._rebuild_post_goal_counts()
        if self._post_possible_track_id == track.track_id:
            self.agent.found_possible_goal = False
            self.agent.possible_goal_temp_gps = np.zeros(2)
            self.agent.goal_map.fill(0)
            self._post_possible_track_id = None
        self.agent.explanation = 'Candidate failed the multi-view contradiction recheck'
        return False

    def _set_post_possible_goal(self, tracks):
        pending = [track for track in tracks if self._post_candidate_is_available(track)
                   and self._candidate_can_be_approached(track)
                   and not self._is_collect_retired(track)
                   and not self._near_collect_retired(track)]
        if pending:
            track = min(pending, key=lambda item: item.current['distance'])
            self.agent.found_possible_goal = True
            self.agent.possible_goal_temp_gps = track.gps.copy()
            self._post_possible_track_id = track.track_id
        elif self._post_possible_track_id in {track.track_id for track in tracks}:
            self.agent.found_possible_goal = False
            self.agent.possible_goal_temp_gps = np.zeros(2)
            self.agent.goal_map.fill(0)
            self._post_possible_track_id = None

    def _record_scenegraph_confirmation_vote(self, track):
        if (instance_confirmation_count_unit() == 'frame'
                and any(step == self.agent.total_steps for _, _, step in track.votes)):
            return  # Reprocessing one observation must not satisfy k=2.
        x = int(self.agent.map_size_cm / 10 - track.gps[1] * 100 / self.agent.resolution)
        y = int(self.agent.map_size_cm / 10 + track.gps[0] * 100 / self.agent.resolution)
        track.votes.append((y, x, self.agent.total_steps))
        self.agent.goal_gps_map[y, x] += 1
        self.agent.goal_gps_timestamp_map[y, x] = self.agent.total_steps
        self._log_confirmation_vote(track)

    def _log_confirmation_vote(self, track):
        from utils.instance_verifier import log_instance_verification
        log_instance_verification('candidate_vote', context={
            'episode': getattr(self.agent, 'count_episodes', None), 'step': self.agent.total_steps,
            'mode': instance_confirmation_mode(), 'stage': 'goal_confirmation'},
            track_id=track.track_id, count=len(track.votes), threshold=instance_confirmation_threshold())

    def _process_scenegraph_post_goals(self, tracks, observations, segment2d_result):
        threshold = instance_confirmation_threshold()
        for track in tracks:
            if (track.current['lock_passed'] and self._post_candidate_is_available(track)
                    and track.current['distance'] < self.agent.distance_threshold):
                self._record_scenegraph_confirmation_vote(track)
        self.agent.found_goal_times = max((len(t.votes) for t in self._positive_evidence.tracks), default=0)
        for track in sorted(tracks, key=lambda item: (-len(item.votes), item.current['distance'])):
            if (not track.current['lock_passed'] or len(track.votes) < threshold
                    or not self._post_candidate_is_available(track)
                    or track.current['distance'] >= self.agent.distance_threshold
                    or not self._candidate_can_be_approached(track)
                    or self._is_collect_retired(track)):
                continue
            if self._post_check_enabled() and not self._post_recheck(track):
                continue
            candidate = track.current
            if self.defer_goal_lock():
                # Same kept [Detection] line the lock would emit, so the box can be labelled offline.
                self._emit_detection('scenegraph', kept=True, category=self.agent.obj_goal,
                                     distance=candidate['distance'], gps=track.gps, center=candidate['center'],
                                     verification_id=getattr(self, '_last_verification_id', None))
                self._collect_retire(track, 'scenegraph_post')
                continue
            self._handle_scenegraph_goal_found(candidate['center'], candidate['direction'], candidate['distance'],
                                               observations, segment2d_result)
            if self.agent.found_goal:
                self.agent.found_goal_times = len(track.votes)
                self.agent.goal_gps = track.gps.copy()
                self._post_goal_track_id = track.track_id
                self._post_possible_track_id = None
                self._emit_goal_lock_once('scenegraph_post', track_id=track.track_id)
                if self.preserve_evidence_on_recovery_enabled():
                    self._recovery_log('candidate_confirmed', track_id=track.track_id,
                                       path='scenegraph', count=len(track.votes), threshold=threshold,
                                       goal_gps=track.gps.tolist(), post_check=self._post_check_enabled())
                return
        self._set_post_possible_goal(tracks)

    def _finalize_post_navigation_goal(self, tracks, threshold):
        # All instance-confirmation modes use the candidate's own votes, whether
        # or not recovery preservation or the post-confirmation check is enabled.
        self.agent.found_goal_times = max((len(t.votes) for t in self._positive_evidence.tracks), default=0)
        for track in sorted(tracks, key=lambda item: (-len(item.votes), item.current['distance'])):
            if (not track.current['lock_passed'] or len(track.votes) < threshold
                    or track.votes[-1][2] != self.agent.total_steps
                    or not self._post_candidate_is_available(track)
                    or track.current['distance'] >= self.agent.distance_threshold
                    or not self._candidate_can_be_approached(track)
                    or self._is_collect_retired(track)):
                continue
            if self._post_check_enabled() and not self._post_recheck(track):
                continue
            if self.defer_goal_lock():
                self._collect_retire(track, 'navigation_confirmed')
                continue
            self.agent.found_goal = True
            self.agent.found_possible_goal = False
            self.agent.found_goal_times = len(track.votes)
            self.agent.goal_gps = track.gps.copy()
            self._post_goal_track_id = track.track_id
            self._post_possible_track_id = None
            self.agent.rag_latest_goal_detection = track.detection
            if self.agent.rag_check_enabled and track.detection is not None:
                self._verify_detection_with_rag()
            if hasattr(self.agent, '_temp_latest_detection'):
                del self.agent._temp_latest_detection
            if self.agent.found_goal:
                self.agent.explanation = f'Goal confirmed from candidate evidence: {self.agent.obj_goal}'
                self._emit_goal_lock_once('navigation_confirmed', track_id=track.track_id)
                if self.preserve_evidence_on_recovery_enabled():
                    self._recovery_log('candidate_confirmed', track_id=track.track_id,
                                       count=len(track.votes), threshold=threshold,
                                       goal_gps=track.gps.tolist(), post_check=self._post_check_enabled())
                return
        self._set_post_possible_goal(tracks)

    def _instance_was_rejected(self, candidate):
        for rejected in self.agent.instance_rejections:
            if candidate['node_id'] is not None and rejected['node_id'] is not None:
                if candidate['node_id'] == rejected['node_id']:
                    return True
            elif self._same_instance_position(candidate['gps'], rejected['gps']):
                return True
        return False

    def _same_instance_position(self, first, second):
        # Existing goal positions are merged in a square of this half-width.
        return (first is not None and second is not None
                and np.max(np.abs(np.asarray(first) - np.asarray(second))) <= self.agent.goal_merge_threshold)

    def _clear_rejected_instance(self, candidate):
        """Remove stale counts/targets for this veto, preserving other candidates."""
        store = getattr(self, '_positive_evidence', None)
        rejected_tracks = set()
        if store is not None:
            for track in store.tracks:
                if self._candidate_matches(candidate, {'node_id': track.node_id, 'gps': track.gps}):
                    track.votes.clear()
                    track.vetoed = True
                    rejected_tracks.add(track.track_id)
            # A graph ID can identify a veto even when this frame has no depth.
            if store.tracks:
                self._rebuild_post_goal_counts()
        gps = candidate['gps']
        if (store is None or not store.tracks) and gps is not None:
            map_x = int(self.agent.map_size_cm/10 - gps[1]*100/self.agent.resolution)
            map_y = int(self.agent.map_size_cm/10 + gps[0]*100/self.agent.resolution)
            radius = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)
            if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:
                region = (slice(max(map_y-radius, 0), min(map_y+radius+1, self.agent.map_size)),
                          slice(max(map_x-radius, 0), min(map_x+radius+1, self.agent.map_size)))
                counts = self.agent.goal_gps_map[region]
                positive = counts > 0
                self.agent.goal_gps_timestamp_map[region][positive] = 0
                counts[positive] = 0
                self.agent.found_goal_times = max(0, self.agent.goal_gps_map.max())

        goal_track = self._post_goal_track_id
        possible_track = self._post_possible_track_id
        clear_goal = self.agent.found_goal and (
            goal_track in rejected_tracks if goal_track is not None
            else self._same_instance_position(self.agent.goal_gps, gps))
        clear_possible = self.agent.found_possible_goal and (
            possible_track in rejected_tracks if possible_track is not None
            else self._same_instance_position(self.agent.possible_goal_temp_gps, gps))
        if clear_goal:
            self.agent.found_goal = False
            self.agent._goal_lock_active = False
            self.agent.goal_gps = np.array([0., 0.])
            self._post_goal_track_id = None
        if clear_possible:
            self.agent.found_possible_goal = False
            self.agent.possible_goal_temp_gps = np.array([0., 0.])
            self._post_possible_track_id = None
        if clear_goal or clear_possible:
            self.agent.goal_map.fill(0)
            self.agent.rag_latest_goal_detection = None
            if hasattr(self.agent, '_temp_latest_detection'):
                del self.agent._temp_latest_detection
            self.agent.explanation = 'Candidate permanently rejected by the configured verification rule'

    def _pixel_direction(self, x):
        """Horizontal angle (deg) of RGB column x; CoIN uses its own RGB intrinsics."""
        geometry = getattr(self.agent, 'coin_geometry', None)
        if geometry is None:
            return (x - 320) * 79 / 640
        return geometry.pixel_direction(x)

    def _depth_search_bounds(self):
        """Exclusive row/column limits of the fallback depth search."""
        geometry = getattr(self.agent, 'coin_geometry', None)
        if geometry is None:
            return 479, 639
        height, width = self.agent.depth.shape[:2]
        return height - 1, width - 1

    def _scenegraph_goal_geometry(self, mask):
        center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
        center_point = torch.tensor([center_point[1], center_point[0]])
        direction = self._pixel_direction(center_point[0])
        distance = self.agent.depth[center_point[1], center_point[0], 0]
        distance = self._fallback_goal_depth(center_point, distance)
        return center_point, direction, distance

    def _navigation_goal_geometry(self, box, original_idx, observations):
        center, _, distance = self.agent.get_unoccluded_center_point(box.to(torch.int64), original_idx, self.agent.depth)
        direction = self._pixel_direction(center[0])
        # Preserve the existing navigation detector's GPS/fallback ordering.
        gps = self.agent.get_goal_gps(observations, direction, distance)
        distance = self._fallback_goal_depth(center, distance)
        return center, direction, distance, gps

    def _fallback_goal_depth(self, center_point, distance):
        k, pos_neg = 0, 1
        max_row, max_col = self._depth_search_bounds()
        while distance >= 100 and 0<center_point[1]+int(pos_neg*k)<max_row and 0<center_point[0]+int(pos_neg*k)<max_col:
            pos_neg *= -1
            k += 0.5
            distance = max(
                self.agent.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                self.agent.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
        return distance

    def _process_navigation_goal_bboxes(self, goal_bbox, goal_bbox_indices, observations, *, post_tracks=None):
        """
        Process goal bboxes from navigation detector.

        Args:
            goal_bbox: List of goal bboxes
            goal_bbox_indices: Original indices in predictions
            observations: Current observations
        """
        possible_goal_detected_before = copy.deepcopy(self.agent.found_possible_goal)
        goal_prediction = copy.deepcopy(self.agent.current_obj_predictions)
        goal_prediction.bbox = torch.stack(goal_bbox)

        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        shortest_distance = 120
        shortest_distance_angle = 0

        for idx_box, box in enumerate(goal_prediction.bbox):
            if post_tracks is not None and not self._post_candidate_is_available(post_tracks[idx_box]):
                continue
            box = box.to(torch.int64)
            original_idx = goal_bbox_indices[idx_box] if idx_box < len(goal_bbox_indices) else idx_box

            center_point, temp_direction, temp_distance, goal_gps = self._navigation_goal_geometry(
                box, original_idx, observations)
            obj_labels = self.agent.current_obj_predictions.get_field("labels")
            detected_label = obj_labels[original_idx] if original_idx < len(obj_labels) else self.agent.obj_goal

            self._emit_detection('navigation', kept=True, box=box.tolist(),
                                 category=detected_label, distance=temp_distance,
                                 gps=goal_gps, center=center_point,
                                 verification_id=getattr(self, '_last_verification_id', None),
                                 prediction_idx=original_idx)

            if temp_distance >= self.agent.distance_threshold:
                # Candidate mode sets the flag and coordinates together, after
                # checking recovery eligibility in _set_post_possible_goal.
                if post_tracks is None:
                    self.agent.found_possible_goal = True
                self.agent.explanation = f"Potential goal detected: {self.agent.obj_goal} at {temp_distance:.2f}m away (beyond {self.agent.distance_threshold}m threshold)"
                far_status = ("仅设置found_possible_goal，不记录GPS" if post_tracks is None
                              else "不计票，接近状态由候选恢复条件决定")
                vprint('detection', f"[距离检查] 检测到 {detected_label}, 距离={temp_distance:.2f}m >= 阈值{self.agent.distance_threshold}m, "
                      f"❌ 太远，{far_status}")
            else:
                # Goal found within threshold
                vprint('detection', f"[距离检查] 检测到 {detected_label}, 距离={temp_distance:.2f}m < 阈值{self.agent.distance_threshold}m, "
                      f"✓ 距离合适，记录GPS位置...")
                self._handle_navigation_goal_found(
                    box, original_idx, goal_gps, temp_distance, observations,
                    post_track=post_tracks[idx_box] if post_tracks is not None else None,
                )

            # Update shortest distance
            if temp_distance < shortest_distance:
                shortest_distance = temp_distance
                shortest_distance_angle = temp_direction

        if post_tracks is not None:
            self._finalize_post_navigation_goal(post_tracks, instance_confirmation_threshold())
            return

        # Update found_goal state based on detection count
        self.agent.found_goal_times = self.agent.goal_gps_map.max()
        detection_threshold = self.agent.get_goal_detection_threshold()

        if len(goal_bbox) > 0:
            print(f"\n[目标计数] goal_gps_map最大计数={int(self.agent.found_goal_times)}/{detection_threshold}, "
                  f"需要>={detection_threshold}次才能确认目标")
            if self.agent.found_goal_times > 0:
                # Show where detections are in the map
                detection_positions = np.where(self.agent.goal_gps_map > 0)
                num_positions = len(detection_positions[0])
                print(f"[目标计数] 检测到{num_positions}个不同位置的目标候选点:")
                for i in range(min(num_positions, 5)):  # Show top 5
                    y, x = detection_positions[0][i], detection_positions[1][i]
                    count = int(self.agent.goal_gps_map[y, x])
                    print(f"  位置{i+1}: map({y},{x}), 计数={count}")
            else:
                print(f"[目标计数] ⚠️ 没有有效的GPS记录（所有检测距离可能都超过阈值）")
        else:
            print(f"[目标计数] 当前帧未检测到目标物体")

        if self.agent.found_goal_times >= detection_threshold:
            self.agent.found_goal = True

        # Update goal GPS if found
        if self.agent.found_goal:
            # Transfer temp detection to latest detection BEFORE RAG verification
            if hasattr(self.agent, '_temp_latest_detection'):
                self.agent.rag_latest_goal_detection = self.agent._temp_latest_detection
                # Don't delete yet - will be cleaned up in _select_most_recent_goal_position

            # RAG verification
            if self.agent.rag_check_enabled and self.agent.rag_latest_goal_detection is not None:
                self._verify_detection_with_rag()

            # Set goal GPS if still found after RAG check
            if self.agent.found_goal:
                self._select_most_recent_goal_position(detection_threshold)
        elif self.agent.found_goal_times > 0:
            self.agent.found_possible_goal = True
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
        elif not possible_goal_detected_before:
            self.agent.possible_goal_temp_gps = self.agent.get_goal_gps(observations, shortest_distance_angle, shortest_distance)

    def _handle_navigation_goal_found(self, box, original_idx, goal_gps, temp_distance, observations, *, post_track=None):
        """
        Handle goal found from navigation detector (including RAG data collection).

        Args:
            box: Goal bbox
            original_idx: Original index in predictions
            goal_gps: Goal GPS coordinates
            temp_distance: Distance to goal
            observations: Current observations
        """
        # Calculate map coordinates (use correct conversion formula)
        map_x = int(self.agent.map_size_cm/10 - goal_gps[1]*100/self.agent.resolution)
        map_y = int(self.agent.map_size_cm/10 + goal_gps[0]*100/self.agent.resolution)

        # Check if this position is marked as false positive (value == -1)
        if 0 <= map_y < self.agent.map_size and 0 <= map_x < self.agent.map_size:
            if self.agent.goal_gps_map[map_y, map_x] == -1:
                print(f"[False Positive Skip] Position map({map_y},{map_x}) is marked as false positive, skipping detection update")
                return  # Skip this detection entirely

        # Update goal_gps_map
        thres = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)

        if post_track is not None and not self._post_candidate_is_available(post_track):
            return
        if (post_track is not None and instance_confirmation_count_unit() == 'frame'
                and any(step == self.agent.total_steps for _, _, step in post_track.votes)):
            return
        if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:

            goal_gps_map_local = self.agent.goal_gps_map[
                max(map_y - thres, 0):
                min(map_y + thres, self.agent.map_size - 1),
                max(map_x - thres, 0):
                min(map_x + thres, self.agent.map_size - 1)
            ]
            timestamp_map_local = self.agent.goal_gps_timestamp_map[
                max(map_y - thres, 0):
                min(map_y + thres, self.agent.map_size - 1),
                max(map_x - thres, 0):
                min(map_x + thres, self.agent.map_size - 1)
            ]

            if goal_gps_map_local.max() > 0:
                # Update existing detection
                max_y, max_x = np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], \
                               np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]
                old_count = int(goal_gps_map_local[max_y, max_x])
                goal_gps_map_local[max_y, max_x] = goal_gps_map_local[max_y, max_x] + 1
                timestamp_map_local[max_y, max_x] = self.agent.total_steps
                vote_y = max(map_y - thres, 0) + int(max_y)
                vote_x = max(map_x - thres, 0) + int(max_x)
                vprint('detection', f"[GPS记录] 目标位置map({map_y},{map_x}), 距离={temp_distance:.2f}m, "
                      f"在{self.agent.goal_merge_threshold:g}m范围内找到已有检测, 计数: {old_count}→{old_count+1}")
            else:
                # First detection at this location
                self.agent.goal_gps_map[map_y, map_x] = 1
                self.agent.goal_gps_timestamp_map[map_y, map_x] = self.agent.total_steps
                vote_y, vote_x = map_y, map_x
                vprint('detection', f"[GPS记录] 目标位置map({map_y},{map_x}), 距离={temp_distance:.2f}m, "
                      f"首次检测到此位置, 计数: 0→1")
            if post_track is not None:
                post_track.votes.append((vote_y, vote_x, self.agent.total_steps))
                self._log_confirmation_vote(post_track)
                if self.preserve_evidence_on_recovery_enabled():
                    self._recovery_log('candidate_vote', track_id=post_track.track_id,
                                       count=len(post_track.votes), map_count=int(self.agent.goal_gps_map[vote_y, vote_x]))

        self.agent.found_possible_goal = False

        # Extract crop for RAG
        box_np = box.cpu().numpy()
        crop = observations["rgb"][int(box_np[1]):int(box_np[3]), int(box_np[0]):int(box_np[2])]

        # Get detected label for debugging
        obj_labels = self.agent.current_obj_predictions.get_field("labels")
        detected_label = obj_labels[original_idx] if original_idx < len(obj_labels) else self.agent.obj_goal

        vprint('rag', f"[RAG Window] Adding detection: detected_label='{detected_label}', using caption='{self.agent.obj_goal}' (goal object)")

        # Get confidence
        original_scores = self.agent.current_obj_predictions.get_field("scores")
        initial_confidence = float(original_scores[original_idx]) if original_idx < len(original_scores) else 0.8
        capped_confidence = min(initial_confidence, 0.9)

        # Update sliding window - CRITICAL: use obj_goal as caption, not detector's label
        # This ensures we only save correct goal object names to the knowledge base
        detection_dict = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detector label
            'confidence': capped_confidence,
            'distance': temp_distance,
            'step': self.agent.total_steps
        }
        self.agent.rag_sliding_window.append(detection_dict)

        # Keep only latest N detections
        if len(self.agent.rag_sliding_window) > self.agent.rag_sliding_window_size:
            self.agent.rag_sliding_window.pop(0)

        # Store for potential latest detection update (after confirmation)
        # Also use obj_goal for consistency
        self.agent._temp_latest_detection = {
            'crop': crop.copy(),
            'caption': self.agent.obj_goal,  # ✅ Use goal name, not detector label
            'confidence': capped_confidence,
            'distance': temp_distance,
            'verified': True,
            'map_x': map_x,  # Store map coordinates for false positive marking
            'map_y': map_y
        }
        if post_track is not None:
            post_track.detection = self.agent._temp_latest_detection

    def _select_most_recent_goal_position(self, detection_threshold):
        """
        Select the most recent goal position from goal_gps_map.

        Args:
            detection_threshold: Minimum detections required
        """
        positions_above_threshold = np.where(self.agent.goal_gps_map >= detection_threshold)

        if len(positions_above_threshold[0]) > 0:
            # Get timestamps for all positions above threshold
            timestamps = self.agent.goal_gps_timestamp_map[positions_above_threshold]

            # Find most recent position
            most_recent_idx = np.argmax(timestamps)
            goal_y_map = positions_above_threshold[0][most_recent_idx]  # row in map
            goal_x_map = positions_above_threshold[1][most_recent_idx]  # col in map

            # Reverse transform from map coordinates to GPS coordinates
            # Forward: map_x = map_size/10 - goal_gps[1]*100/resolution
            # Reverse: goal_gps[0] = (map_y - map_size/10) * resolution / 100
            #          goal_gps[1] = (map_size/10 - map_x) * resolution / 100
            goal_gps_0 = (goal_y_map - self.agent.map_size_cm/10) * self.agent.resolution / 100
            goal_gps_1 = (self.agent.map_size_cm/10 - goal_x_map) * self.agent.resolution / 100

            self.agent.goal_gps = np.array([goal_gps_0, goal_gps_1], dtype=float)
            print(f"[Goal Selection] Selected most recent detection at map position ({goal_y_map}, {goal_x_map}), "
                  f"count={int(self.agent.goal_gps_map[goal_y_map, goal_x_map])}, "
                  f"last_seen_step={int(self.agent.goal_gps_timestamp_map[goal_y_map, goal_x_map])}, "
                  f"threshold={detection_threshold}")
        else:
            # Fallback: use max count
            max_positions = np.where(self.agent.goal_gps_map == self.agent.goal_gps_map.max())
            goal_y_map = max_positions[0][0]
            goal_x_map = max_positions[1][0]

            # Reverse transform
            goal_gps_0 = (goal_y_map - self.agent.map_size_cm/10) * self.agent.resolution / 100
            goal_gps_1 = (self.agent.map_size_cm/10 - goal_x_map) * self.agent.resolution / 100
            self.agent.goal_gps = np.array([goal_gps_0, goal_gps_1], dtype=float)
            print(f"[Goal Selection] WARNING: No position meets threshold, using max count fallback at map({goal_y_map}, {goal_x_map})")

        self.agent.explanation = f"Goal found: {self.agent.obj_goal} (detection #{int(self.agent.found_goal_times)})"

        self._emit_goal_lock_once('navigation_count')

        # Clean up temp detection (already transferred to rag_latest_goal_detection before RAG check)
        if hasattr(self.agent, '_temp_latest_detection'):
            delattr(self.agent, '_temp_latest_detection')

    def _verify_detection_with_rag(self):
        """
        Verify current goal detection using RAG manager.
        May reset goal detection if RAG rejects the detection.
        """
        if is_instance_dataset(self.agent.dataset) and not instance_confirmation_enabled():
            return
        is_verified, adjusted_conf, explanation, comparison_doc = self.agent.rag_manager.verify_detection(
            obj_category=self.agent.obj_goal,
            crop=self.agent.rag_latest_goal_detection['crop'],
            caption=self.agent.rag_latest_goal_detection['caption'],
            confidence=self.agent.rag_latest_goal_detection['confidence']
        )

        # Store comparison document for visualization
        self.agent.rag_comparison_doc = comparison_doc

        # Update confidence based on RAG verification
        self.agent.rag_latest_goal_detection['confidence'] = adjusted_conf
        self.agent.rag_latest_goal_detection['verified'] = is_verified

        print(f"[RAG Check] {explanation}")

        # If RAG rejects, check adjusted confidence against threshold
        if not is_verified:
            # Special handling for false positive rejection
            if adjusted_conf == 0.0:
                print(f"[RAG Check] 🚫 FALSE POSITIVE DETECTED! Resetting goal detection and re-entering FBE.")

                # Mark this position on goal_gps_map as false positive (-1) to prevent future detections
                if 'map_x' in self.agent.rag_latest_goal_detection and 'map_y' in self.agent.rag_latest_goal_detection:
                    map_x = self.agent.rag_latest_goal_detection['map_x']
                    map_y = self.agent.rag_latest_goal_detection['map_y']

                    if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:
                        # Mark this position and nearby area as false positive
                        thres = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)
                        y_start = max(map_y - thres, 0)
                        y_end = min(map_y + thres + 1, self.agent.map_size)
                        x_start = max(map_x - thres, 0)
                        x_end = min(map_x + thres + 1, self.agent.map_size)

                        self.agent.goal_gps_map[y_start:y_end, x_start:x_end] = -1
                        print(f"[False Positive Map] Marked position map({map_y},{map_x}) and nearby area "
                              f"({y_end-y_start}x{x_end-x_start} pixels) as false positive (value=-1)")

                self.agent.reset_goal_detection(reason='rag_rejected')
            else:
                # Regular rejection - check against threshold
                detector_threshold = self.agent.get_detector_confidence_threshold()
                if adjusted_conf < detector_threshold:
                    print(f"[RAG Check] ❌ Detection rejected: adjusted conf {adjusted_conf:.3f} < threshold {detector_threshold:.3f}")
                    self.agent.reset_goal_detection(reason='rag_rejected')
                else:
                    print(f"[RAG Check] ⚠️  RAG rejected but keeping goal: adjusted conf {adjusted_conf:.3f} >= threshold {detector_threshold:.3f}")
        else:
            print(f"[RAG Check] ✓ Detection accepted (adjusted conf: {adjusted_conf:.3f})")

    def _get_glip_real_label(self, prediction):
        """
        Convert GLIP prediction indices to string labels.

        Args:
            prediction: GLIP prediction object

        Returns:
            List of string labels
        """
        labels = prediction.get_field("labels").tolist()
        new_labels = []

        # Check if labels are already strings (FastSAM) or numeric indices (GLIP)
        if labels and isinstance(labels[0], str):
            # FastSAM already returns string labels, no conversion needed
            return labels

        # GLIP returns numeric indices, convert to string labels
        if self.agent.glip_demo and self.agent.glip_demo.entities and self.agent.glip_demo.plus:
            for i in labels:
                if 0 <= i - self.agent.glip_demo.plus < len(self.agent.glip_demo.entities):
                    new_labels.append(self.agent.glip_demo.entities[i - self.agent.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]

        return new_labels

    def _save_detection_images(self, image_bgr, observations, before_detection=True,
                               bboxes=None, labels=None, scores=None):
        """
        Save detection images (before and after).

        Args:
            image_bgr: Image in BGR format
            observations: Current observations
            before_detection: If True, save original image; if False, save with bboxes
            bboxes: Bounding boxes (numpy array)
            labels: Object labels (list of strings)
            scores: Confidence scores (numpy array)
        """
        if not self.agent.args.visualize:
            return
        import cv2
        import os
        from pathlib import Path

        # Create output directory
        output_dir = Path(f"data/glip_detections/episode_{self.agent.count_episodes}")
        output_dir.mkdir(parents=True, exist_ok=True)

        step = self.agent.total_steps

        if before_detection:
            # Save original image
            save_path = output_dir / f"step_{step:04d}_before.jpg"
            cv2.imwrite(str(save_path), image_bgr)

        else:
            # Draw bboxes and save
            image_with_bbox = image_bgr.copy()

            # Define colors for different categories (BGR format)
            np.random.seed(42)  # Fixed seed for consistent colors
            colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(100)]

            for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                x1, y1, x2, y2 = bbox.astype(int)
                color = colors[i % len(colors)]

                # Draw rectangle
                cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, 2)

                # Prepare label text
                label_text = f"{label}: {score:.2f}"

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    image_with_bbox,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    image_with_bbox,
                    label_text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1,
                    cv2.LINE_AA
                )

            # Save image with bboxes
            save_path = output_dir / f"step_{step:04d}_after.jpg"
            cv2.imwrite(str(save_path), image_with_bbox)
