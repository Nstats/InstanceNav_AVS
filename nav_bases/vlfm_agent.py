"""VLFM (Yokoyama et al., ICRA 2024) as an object-navigation base, with optional AVS.

The official VLFM code (``VLFM_ROOT``, default /data/lxj/vlfm) supplies the policy:
ITMPolicyV2 (BLIP-2 value map + frontier choice), YOLOv7 / GroundingDINO detection,
MobileSAM masks, the object point-cloud map and the PointNav controller. Only the
habitat glue is replaced, because this repository runs the bundled habitat-lab 0.2.1:

* observations come from our environment as numpy arrays. Depth (metres) is mapped to
  VLFM's sensor convention, 0.5-5.0 m normalised to [0, 1], exactly what the official
  HM3D config gives the policy, so the pretrained PointNav weights see their own input;
* the PointNav checkpoint is loaded as a plain state_dict (tools/export_vlfm_pointnav.py);
* the agent exposes the PSG_Nav_Agent interface that habitat.core.benchmark uses.

CoIN (configs/coin_contextnav.yaml) differs from InstanceNav only in the sensors:
RGB 512x512 at 42 degrees, depth 512x512 at 79 degrees (0-5 m), both at 1.31 m. The
obstacle map and PointNav use the full depth view with its own intrinsics; the object
and value maps pair each RGB pixel with depth resampled onto the RGB grid
(utils/coin_geometry.CoinGeometry) and use the RGB intrinsics, so boxes and masks land
on the right depth. Depth below 0.5 m (including CoIN's 0 m holes) reads 0 after
normalisation, exactly as the official 0.5-5 m sensor reports it. The episode's
category text is the detector target (YOLOv7 for COCO classes, else GroundingDINO with
that text as its only class); the full CoIN instruction is used only by AVS.

ENABLE_INSTANCE_CONFIRMATION=false runs VLFM unchanged (category detection decides the
goal). true adds AVS (nav_bases/avs_decision.py): only a detection the VLM confirms
from within 4 m may enter the target map. Exploration (value map prompt, frontiers) is
VLFM's in both cases. Nothing here is imported by PSG_Nav.py.
"""
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VLFM_ROOT = Path(os.environ.get('VLFM_ROOT', '/data/lxj/vlfm'))
VLFM_DEPS = Path(os.environ.get('VLFM_DEPS', str(VLFM_ROOT / '.vlfm-deps')))
POINTNAV_WEIGHTS = Path(os.environ.get('VLFM_POINTNAV_WEIGHTS',
                                       str(ROOT / 'data/vlfm/pointnav_weights_state_dict.pth')))
# Official VLFM HM3D sensor range (config/experiments/vlfm_objectnav_hm3d.yaml via
# habitat objectnav_hm3d): the policy and its maps were built for normalised 0.5-5 m.
VLFM_MIN_DEPTH_M = 0.5
VLFM_MAX_DEPTH_M = 5.0
# VLFMConfig defaults as resolved in the official HM3D run (policy section of its config).
VLFM_POLICY_DEFAULTS = dict(
    text_prompt='Seems like there is a target_object ahead.',
    depth_image_shape=(224, 224), pointnav_stop_radius=0.9, use_max_confidence=False,
    object_map_erosion_size=5, obstacle_map_area_threshold=1.5, min_obstacle_height=0.61,
    max_obstacle_height=0.88, hole_area_thresh=100000, use_vqa=False, vqa_prompt='Is this ',
    coco_threshold=0.8, non_coco_threshold=0.4, agent_radius=0.18,
)
# InstanceNav (HM3D) goal names -> VLFM's HM3D_ID_TO_NAME detector classes.
INSTANCENAV_TO_VLFM = {'chair': 'chair', 'bed': 'bed', 'plant': 'potted plant', 'toilet': 'toilet',
                       'tv_monitor': 'tv', 'sofa': 'couch'}
SERVER_PORTS = {'GROUNDING_DINO_PORT': 12181, 'BLIP2ITM_PORT': 12182, 'SAM_PORT': 12183,
                'YOLOV7_PORT': 12184}


def import_vlfm():
    """Make the official VLFM package importable next to habitat-lab 0.2.1.

    VLFM and its vendored dependencies are appended after this repository's paths, so
    habitat (0.2.1) and every package already in the environment win. habitat_baselines
    is blocked: VLFM then uses its own non-habitat PointNav code path.
    """
    if 'habitat_baselines' not in sys.modules:
        sys.modules['habitat_baselines'] = None
    for path in (str(VLFM_ROOT), str(VLFM_DEPS)):
        if path not in sys.path:
            sys.path.append(path)
    import habitat
    if not habitat.__file__.startswith(str(ROOT)):
        raise RuntimeError(f'habitat must come from this repository, got {habitat.__file__}')
    import vlfm.policy.utils.pointnav_policy as pointnav_policy
    from nav_bases.vlfm_pointnav import load_discrete_pointnav_policy
    # WrappedPointNavResNetPolicy resolves this name at construction (this process only).
    pointnav_policy.load_pointnav_policy = load_discrete_pointnav_policy
    from vlfm.policy.itm_policy import ITMPolicyV2
    return ITMPolicyV2


def check_servers():
    missing = []
    for name, default in SERVER_PORTS.items():
        port = int(os.environ.get(name, default))
        with socket.socket() as probe:
            probe.settimeout(2)
            if probe.connect_ex(('localhost', port)) != 0:
                missing.append(f'{name}={port}')
    if missing:
        raise RuntimeError('VLFM model servers are not listening (' + ', '.join(missing)
                           + '); start them with scripts/launch_vlfm_servers.sh')


def vlfm_target_name(dataset, category):
    if dataset == 'instancenav':
        if category not in INSTANCENAV_TO_VLFM:
            raise ValueError(f'No VLFM detector class for InstanceNav category {category!r}')
        return INSTANCENAV_TO_VLFM[category]
    # Open-vocabulary datasets (CoIN): the episode's category text is the detector prompt.
    return category.replace('_', ' ').strip()


def normalise_depth(depth_m):
    """Metric depth -> VLFM's normalised depth (habitat clips to [0.5, 5] then scales)."""
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = depth.reshape(depth.shape[:2])
    depth = np.where(np.isfinite(depth), depth, VLFM_MIN_DEPTH_M)
    depth = np.clip(depth, VLFM_MIN_DEPTH_M, VLFM_MAX_DEPTH_M)
    return (depth - VLFM_MIN_DEPTH_M) / (VLFM_MAX_DEPTH_M - VLFM_MIN_DEPTH_M)


def validate_avs_settings():
    """The VLFM base implements A0 AVS only; refuse settings it would silently ignore."""
    from configs.navigation_config import (
        candidate_approach_enabled, instance_collect_only_enabled, instance_confirmation_enabled,
        instance_confirmation_mode, instance_confirmation_threshold, instance_rejection_policy,
        post_confirmation_contradiction_check_enabled, relative_evidence_enabled,
        validate_instance_ablation, verify_before_candidate_enabled)
    summary = validate_instance_ablation()
    if not instance_confirmation_enabled():
        return dict(summary, avs=False)
    problems = []
    if instance_confirmation_mode() != 'positive_match':
        problems.append('INSTANCE_CONFIRMATION_MODE=positive_match')
    if instance_confirmation_threshold() != 1:
        problems.append('INSTANCE_CONFIRMATION_THRESHOLD=1')
    if candidate_approach_enabled():
        problems.append('ENABLE_CANDIDATE_APPROACH=false (A0)')
    if instance_rejection_policy() != 'none':
        problems.append('INSTANCE_REJECTION_POLICY=none')
    if verify_before_candidate_enabled():
        problems.append('VERIFY_BEFORE_CANDIDATE=false')
    if (instance_collect_only_enabled() or relative_evidence_enabled()
            or post_confirmation_contradiction_check_enabled()):
        problems.append('collect-only, relative evidence and post check disabled')
    if problems:
        raise ValueError('NAV_BASE=vlfm with instance confirmation needs ' + '; '.join(problems))
    return dict(summary, avs=True)
def build_policy_class():
    """ITMPolicyV2 with VLFM's HabitatMixin behaviour, for numpy observations."""
    ITMPolicyV2 = import_vlfm()
    from depth_camera_filtering import filter_depth
    from vlfm.mapping.object_point_cloud_map import too_offset
    from vlfm.utils.geometry_utils import get_fov, transform_points, xyz_yaw_to_tf_matrix

    from nav_bases.avs_decision import Proposal

    class VLFMPolicy(ITMPolicyV2):
        _stop_action = torch.tensor([[0]], dtype=torch.long)
        _turn_left = torch.tensor([[2]], dtype=torch.long)

        def __init__(self, *, camera_height, rgb_fov_deg, depth_fov_deg, rgb_width, depth_width,
                     rgb_pixel_map=None, **kwargs):
            super().__init__(**kwargs)
            self._camera_height = float(camera_height)
            self._min_depth, self._max_depth = VLFM_MIN_DEPTH_M, VLFM_MAX_DEPTH_M
            # Obstacle map and PointNav use the depth camera; the object and value maps
            # pair RGB with depth on the RGB pixel grid (identical for InstanceNav).
            self._camera_fov = np.deg2rad(depth_fov_deg)
            self._fx = self._fy = depth_width / (2 * np.tan(self._camera_fov / 2))
            self._rgb_fov = np.deg2rad(rgb_fov_deg)
            self._rgb_fx = rgb_width / (2 * np.tan(self._rgb_fov / 2))
            self._rgb_pixel_map = rgb_pixel_map  # (rows, cols) into depth, or None
            self.avs = None            # AVSDecision when instance confirmation is on
            self.avs_context = dict    # callable -> log context of the current step

        # --- HabitatMixin, unchanged in behaviour -------------------------------------
        def _initialize(self):
            """Turn left 30 degrees 12 times to get a 360 view at the beginning."""
            self._done_initializing = not self._num_steps < 11
            return self._turn_left

        def _infer_depth(self, rgb, min_depth, max_depth):
            raise NotImplementedError('VLFM infers depth only without a depth sensor')

        def _cache_observations(self, observations):
            if len(self._observations_cache) > 0:
                return
            rgb = observations['rgb']
            nav_depth = normalise_depth(observations['depth'])
            x, y = observations['gps']
            camera_yaw = float(np.asarray(observations['compass']).reshape(-1)[0])
            depth = filter_depth(nav_depth.copy(), blur_type=None)
            # Habitat GPS makes west negative, so flip y
            camera_position = np.array([x, -y, self._camera_height])
            robot_xy = camera_position[:2]
            tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

            self._obstacle_map.update_map(depth, tf_camera_to_episodic, self._min_depth,
                                          self._max_depth, self._fx, self._fy, self._camera_fov)
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)

            if self._rgb_pixel_map is None:
                rgb_depth, rgb_fx, rgb_fov = depth, self._fx, self._camera_fov
            else:
                rows, cols = self._rgb_pixel_map
                rgb_depth = np.ascontiguousarray(depth[rows[:, None], cols[None, :]])
                rgb_fx, rgb_fov = self._rgb_fx, self._rgb_fov
            self._observations_cache = {
                'frontier_sensor': frontiers,
                'nav_depth': torch.from_numpy(nav_depth[None, :, :, None]).to('cuda'),
                'robot_xy': robot_xy,
                'robot_heading': camera_yaw,
                'object_map_rgbd': [(rgb, rgb_depth, tf_camera_to_episodic, self._min_depth,
                                     self._max_depth, rgb_fx, rgb_fx)],
                'value_map_rgbd': [(rgb, rgb_depth, tf_camera_to_episodic, self._min_depth,
                                    self._max_depth, rgb_fov)],
            }

        # --- AVS: which detections may enter the target map ----------------------------
        def _update_object_map(self, rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy):
            if self.avs is None:
                return super()._update_object_map(rgb, depth, tf_camera_to_episodic,
                                                  min_depth, max_depth, fx, fy)
            detections = self._get_object_detections(rgb)
            height, width = rgb.shape[:2]
            self._object_masks = np.zeros((height, width), dtype=np.uint8)
            if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
                depth = self._infer_depth(rgb, min_depth, max_depth)
            proposals, masks = [], []
            for idx in range(len(detections.logits)):
                box = detections.boxes[idx] * np.array([width, height, width, height])
                mask = self._mobile_sam.segment_bbox(rgb, box.tolist())
                masks.append(mask)
                proposals.append(self._proposal(box, mask, depth, tf_camera_to_episodic,
                                                min_depth, max_depth, fx, fy))
            for idx in self.avs.select(proposals, rgb, self.avs_context()):
                self._object_masks[masks[idx] > 0] = 1
                cloud = proposals[idx].cloud
                if self._target_object in self._object_map.clouds:
                    cloud = np.concatenate((self._object_map.clouds[self._target_object], cloud), axis=0)
                self._object_map.clouds[self._target_object] = cloud
            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)
            if self.avs.is_committed and not self._object_map.has_object(self._target_object):
                self.avs.withdraw('target_map_cleared', self.avs_context())
            return detections

        def _proposal(self, box, mask, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy):
            """ObjectPointCloudMap.update_map up to the point where it stores the cloud."""
            object_map = self._object_map
            local_cloud = object_map._extract_object_cloud(depth, mask, min_depth, max_depth, fx, fy)
            if len(local_cloud) == 0:
                return Proposal(box, None, None)
            if too_offset(mask):
                within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
            else:
                within_range = ((local_cloud[:, 0] <= max_depth * 0.95) * 1.0).astype(np.float32)
                within_range[within_range == 0] = np.random.rand()
            global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
            global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)
            curr_position = tf_camera_to_episodic[:3, 3]
            closest_point = object_map._get_closest_point(global_cloud, curr_position)
            dist = float(np.linalg.norm(closest_point[:3] - curr_position))
            if dist < 1.0:
                # Object is too close to trust as a valid object (as update_map)
                return Proposal(box, None, dist)
            return Proposal(box, global_cloud, dist)

    return VLFMPolicy
class VLFMNavAgent:
    """PSG_Nav_Agent's evaluation interface around the VLFM policy."""

    nav_base = 'vlfm'

    def __init__(self, task_config, args=None):
        from configs.dataset_categories import dataset_to_unified_name
        from utils.log_utils import code_version, log_event
        from utils.runtime_config import config_summary

        self.config = task_config
        self.args = args
        self.dataset = getattr(args, 'dataset', None) or 'hssd'
        self.dataset_to_unified_name = dataset_to_unified_name
        self.max_episode_steps = int(task_config.ENVIRONMENT.MAX_EPISODE_STEPS)
        if not bool(getattr(task_config.ENVIRONMENT, 'ATOMIC_ACTIONS', False)):
            raise ValueError('The VLFM base counts atomic actions; set ENVIRONMENT.ATOMIC_ACTIONS True')
        self.avs_settings = validate_avs_settings()
        self.distance_threshold = 4
        sim = task_config.SIMULATOR
        rgb, depth = sim.RGB_SENSOR, sim.DEPTH_SENSOR
        if bool(getattr(depth, 'NORMALIZE_DEPTH', False)):
            raise ValueError('The VLFM base needs metric depth (DEPTH_SENSOR.NORMALIZE_DEPTH False)')
        rgb_pixel_map = None
        if self.dataset == 'coin':
            from utils.coin_geometry import CoinGeometry
            geometry = CoinGeometry.from_config(task_config)
            rgb_pixel_map = (geometry.aligned_rows, geometry.aligned_cols)
        elif (float(rgb.HFOV), int(rgb.WIDTH), int(rgb.HEIGHT)) != (
                float(depth.HFOV), int(depth.WIDTH), int(depth.HEIGHT)):
            raise ValueError('RGB and depth cameras differ; only CoIN has a resampling path')
        toggles = ['ENABLE_INSTANCE_CONFIRMATION', 'INSTANCE_CONFIRMATION_THRESHOLD',
                   'INSTANCE_CONFIRMATION_MODE', 'INSTANCE_VERIFICATION_PROMPT',
                   'INSTANCE_VERIFICATION_SCOPE', 'INSTANCE_REJECTION_POLICY',
                   'ENABLE_CANDIDATE_APPROACH', 'INSTANCE_VERIFICATION_RECORD_INPUTS',
                   'INSTANCE_ABLATION_ID', 'INSTANCE_ABLATION_MANIFEST', 'NAV_BASE']
        log_event('Run Header', code_version=code_version(), nav_base=self.nav_base,
                  vlfm_root=str(VLFM_ROOT), pointnav_weights=str(POINTNAV_WEIGHTS),
                  avs=self.avs_settings, max_episode_steps=self.max_episode_steps,
                  habitat_config=config_summary(task_config),
                  toggles={name: os.environ[name] for name in toggles if name in os.environ})

        if not POINTNAV_WEIGHTS.is_file():
            raise FileNotFoundError(f'{POINTNAV_WEIGHTS} missing; run tools/export_vlfm_pointnav.py')
        check_servers()
        policy_cls = build_policy_class()
        # The official policy values on every benchmark: on InstanceNav (0.88 m, 0.18 m
        # agent) they equal the agent's own size; CoIN's 1.41 m / 0.17 m agent keeps
        # VLFM's 0.61-0.88 m obstacle band and 0.18 m radius, as Context-Nav's VLFM does.
        settings = dict(VLFM_POLICY_DEFAULTS)
        self.policy = policy_cls(
            pointnav_policy_path=str(POINTNAV_WEIGHTS), visualize=False, compute_frontiers=True,
            camera_height=float(depth.POSITION[1]), rgb_fov_deg=float(rgb.HFOV),
            depth_fov_deg=float(depth.HFOV), rgb_width=int(rgb.WIDTH), depth_width=int(depth.WIDTH),
            rgb_pixel_map=rgb_pixel_map, **settings)
        self.model_client = None
        if self.avs_settings['avs']:
            from nav_bases.avs_decision import AVSDecision
            from utils.instance_verifier import log_instance_verification
            from utils.model_client import get_model_client
            self.model_client = get_model_client(getattr(args, 'server_port', None))
            self.policy.avs = AVSDecision(self._verify, distance_threshold=self.distance_threshold,
                                          log=log_instance_verification)
            self.policy.avs_context = self._log_context
        print(f'[VLFM base] avs={self.avs_settings["avs"]} settings={settings} '
              f'rgb_pixel_map={"coin" if rgb_pixel_map is not None else "identity"}', flush=True)

        self.experiment_name = 'experiment_0'
        worker_gpu = getattr(args, 'worker_gpu', None)
        if getattr(args, 'work_queue', None) and worker_gpu is not None:
            self.experiment_name = f'gpu{int(worker_gpu)}'
        if getattr(args, 'split_l', -1) >= 0:
            self.experiment_name += f'/[{args.split_l}:{args.split_r}]'
        from datetime import datetime
        self.timestamp = getattr(args, 'timestamp', None) or datetime.now().strftime('%m%d_%H%M')

        self.count_episodes = -1
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.found_possible_goal = False
        self.distance_to_goal = None
        self.full_pose = None
        self.random_nav_periods = []
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

    # --- verification -------------------------------------------------------------
    def _log_context(self):
        return {'episode': self.count_episodes, 'step': self.total_steps,
                'category': self.obj_goal, 'mode': 'instance_match', 'nav_base': self.nav_base}

    def _verify(self, rgb, boxes, context):
        from utils.instance_verifier import verify_candidate_decisions
        return verify_candidate_decisions(self.model_client, rgb, boxes, self.obj_goal,
                                          self.goal_attributes, log_context=context)

    # --- benchmark interface --------------------------------------------------------
    def reset(self):
        from utils.model_client import snapshot_api_timing
        episode = self.simulator._env.current_episode
        self.dataset_goal_category = episode.object_category
        self.goal_attributes = getattr(episode, 'goal_attributes', None)
        self.obj_goal = self.dataset_to_unified_name(episode.object_category, self.dataset)
        self.vlfm_target = vlfm_target_name(self.dataset, episode.object_category)
        if self.avs_settings['avs'] and not self.goal_attributes:
            raise ValueError('AVS needs the episode goal description (goal_attributes)')
        self.count_episodes += 1
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.found_possible_goal = False
        self.distance_to_goal = None
        self.random_nav_periods = []
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        if self.policy.avs is not None:
            self.policy.avs.reset()
        # Open-vocabulary targets go to GroundingDINO with the target as the only class.
        self.policy._non_coco_caption = self.vlfm_target + ' .'
        scene = Path(episode.scene_id).name.replace('.basis.glb', '').replace('.glb', '')
        print(f'[Episode {self.count_episodes}] nav_base=vlfm dataset={self.dataset} scene={scene} '
              f'goal={self.obj_goal} vlfm_target={self.vlfm_target!r}', flush=True)
        self._api_timing_baseline = snapshot_api_timing()

    def act(self, observations):
        if self.total_steps >= self.max_episode_steps:
            return {'action': 0}
        masks = torch.tensor([[self.total_steps > 0]], dtype=torch.bool, device='cuda')
        self.total_steps += 1
        obs = {'rgb': observations['rgb'], 'depth': observations['depth'],
               'gps': np.asarray(observations['gps'], dtype=np.float32),
               'compass': observations['compass'], 'objectgoal': self.vlfm_target}
        try:
            action, _ = self.policy.act(obs, None, None, masks, deterministic=True)
        except StopIteration:
            action = self.policy._stop_action
        action = int(torch.as_tensor(action).reshape(-1)[0].item())
        info = self.policy._policy_info
        self.found_goal = bool(info.get('target_detected', False))
        if self.found_goal:
            self.found_goal_times += 1
        rho_theta = info.get('rho_theta')
        self.distance_to_goal = (float(rho_theta[0]) if self.found_goal and rho_theta is not None
                                 else None)
        x, y = obs['gps']
        heading = float(np.asarray(obs['compass']).reshape(-1)[0])
        self.full_pose = np.array([x, y, np.rad2deg(heading)])
        return {'action': action}

    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        if 'success' in metrics:
            self.metrics['success'] = metrics['success']

    def api_timing_delta(self):
        from utils.model_client import diff_api_timing, snapshot_api_timing
        current = snapshot_api_timing()
        baseline = getattr(self, '_api_timing_baseline', None)
        return diff_api_timing(current if baseline is None else baseline, current)
