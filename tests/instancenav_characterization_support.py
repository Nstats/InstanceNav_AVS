"""Fingerprints of InstanceNav-path behaviour, recorded before the CoIN integration.

Every function here calls only APIs that existed before CoIN-Bench support was
added, so the same fingerprints can be computed on the old and the new code.
tests/test_instancenav_characterization.py asserts that they are unchanged.
"""
import contextlib
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import torch

INSTANCENAV_CONFIG = 'configs/instancenav_val20_qwen.yaml'
# Context-free env switches touched by the gate fingerprints.
GATE_ENV = ('ENABLE_INSTANCE_CONFIRMATION', 'ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL',
            'INSTANCE_CONFIRMATION_MODE', 'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK',
            'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'INSTANCE_COLLECT_ONLY',
            'INSTANCE_CONFIRMATION_THRESHOLD', 'ENABLE_RELATIVE_EVIDENCE')


@contextlib.contextmanager
def env(**values):
    saved = {name: os.environ.get(name) for name in GATE_ENV}
    try:
        for name in GATE_ENV:
            os.environ.pop(name, None)
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def binary_fingerprint(tensor):
    array = tensor.detach().cpu().numpy()
    assert set(np.unique(array)).issubset({0.0, 1.0})
    packed = np.packbits(array.astype(bool).ravel())
    return [list(array.shape), int(array.sum()), hashlib.sha256(packed.tobytes()).hexdigest()[:16]]


def instancenav_config():
    import habitat
    return habitat.get_config(INSTANCENAV_CONFIG)


def mapping_agent(config):
    return SimpleNamespace(device=torch.device('cpu'), config=config, map_size_cm=4000)


def synthetic_depth(height=480, width=640):
    """Depth in metres as Habitat returns it (H, W); covers every validity branch."""
    depth = np.tile(np.linspace(1.0, 6.0, width, dtype=np.float32), (height, 1))
    depth[200:300, 250:400] = 1.5          # near obstacle in front
    depth[0:40, :] = 0.0                    # holes
    depth[40:60, :] = 10.0                  # clipped at MAX_DEPTH
    depth[60:70, 100:200] = 0.5             # exactly MIN_DEPTH
    depth[400:420, 500:600] = 9.5           # far, inside range
    return depth


def wall_depth(height=480, width=640, distance=1.2):
    """A frontal wall: dense columns, so the level-camera map is non-empty too."""
    depth = np.full((height, width), distance, dtype=np.float32)
    depth[:, :80] = 3.0
    return depth


def slope_depth(height=480, width=640):
    """Depth varying with row, so obstacles span several cells and survive erosion."""
    rows = np.linspace(3.0, 1.0, height, dtype=np.float32)[:, None]
    depth = np.repeat(rows, width, axis=1)
    depth[:, 500:] += 1.5
    return depth


def mapping_fingerprints():
    """Construction parameters and forward outputs of the three map modules."""
    from utils.utils_fmm.mapping import Semantic_Mapping
    agent = mapping_agent(instancenav_config())
    modules = {
        'sem': Semantic_Mapping(agent),
        'free': Semantic_Mapping(agent, max_height=10, min_height=-150),
        'room': Semantic_Mapping(agent, max_height=200, min_height=-10, num_cats=9),
    }
    inputs = {'ramp': torch.from_numpy(synthetic_depth()), 'wall': torch.from_numpy(wall_depth()),
              'slope': torch.from_numpy(slope_depth())}
    pose = torch.tensor([20.0, 20.0, 37.0])
    type_mask = torch.zeros(9, 480, 640)
    type_mask[0, 150:350, 200:450] = 1
    type_mask[3, 300:480, 0:300] = 1
    score_vec = torch.zeros(9)
    score_vec[0], score_vec[3] = .8, .6
    out = {}
    for name, module in modules.items():
        cm = module.camera_matrix
        record = {'screen': [module.screen_h, module.screen_w], 'fov': float(module.fov),
                  'agent_height': float(module.agent_height),
                  'z_consider': [float(module.min_z_consider), float(module.max_z_consider)],
                  'depth_cm': [module.min_depth_cm, module.max_depth_cm],
                  'camera_matrix': [float(cm.xc), float(cm.zc), round(float(cm.f), 9)]}
        for angle in (0, 30):
            module.set_view_angles(angle)
            for label, depth in inputs.items():
                if name == 'room':
                    last = torch.zeros(1, 9, 800, 800)
                    result = module(depth, pose, last, type_mask, score_vec)
                else:
                    last = torch.zeros(1, 1, 800, 800)
                    result = module(depth, pose, last)
                record[f'{label}_{angle}'] = binary_fingerprint(result)
        out[name] = record
    return out


def stop_rule_fingerprints():
    """(stg, replan, stop) from the real _get_stg on fixed maps."""
    from PSG_Nav import PSG_Nav_Agent
    grid = np.ones((60, 60))
    grid[5:55, 28:32] = 0          # wall with gaps at both ends
    traversible = np.ones((62, 62))
    traversible[1:61, 1:61] = grid
    with_stop = SimpleNamespace(TASK=SimpleNamespace(STOP_DISTANCE=1.0))
    without_stop = SimpleNamespace(TASK=SimpleNamespace())
    cases = [  # name, config, start, goal cells, goal_found
        ('far_open', with_stop, (10, 10), [(50, 12)], True),
        ('within_15_cells', with_stop, (10, 10), [(25, 10)], True),
        ('beyond_25_cells', with_stop, (10, 10), [(35, 10)], True),
        ('behind_wall', with_stop, (30, 20), [(30, 40)], True),
        ('goal_in_wall', with_stop, (30, 18), [(30, 29)], True),
        ('no_stop_distance', without_stop, (10, 10), [(25, 10)], True),
        ('frontier_goal', with_stop, (10, 10), [(18, 10)], False),
        ('multi_goal', with_stop, (10, 10), [(40, 10), (41, 10), (45, 45)], True),
        ('adjacent', with_stop, (10, 10), [(10, 11)], True),
    ]
    out = {}
    for name, config, start, cells, goal_found in cases:
        agent = PSG_Nav_Agent.__new__(PSG_Nav_Agent)
        agent.dilation_deg = 0
        agent.config = config
        goal = np.zeros((60, 60))
        for cell in cells:
            goal[cell] = 1
        (stg_y, stg_x), replan, stop = agent._get_stg(traversible.copy(), list(start), goal, goal_found)
        out[name] = [float(stg_y), float(stg_x), bool(replan), bool(stop)]
    return out


def detection_manager(**agent_fields):
    from utils.object_detection import ObjectDetectionManager
    manager = ObjectDetectionManager.__new__(ObjectDetectionManager)
    manager.agent = SimpleNamespace(**agent_fields)
    return manager


def fallback_depth_fingerprints():
    depth = np.full((480, 640, 1), 3.0, dtype=np.float32)
    depth[220:260, 300:340, 0] = 150.0
    depth[0:12, 0:12, 0] = 150.0
    manager = detection_manager(depth=depth)
    cases = {'in_box': ([320, 240], 150.0), 'valid_start': ([100, 100], 3.0),
             'corner': ([5, 5], 150.0)}
    return {name: float(manager._fallback_goal_depth(center, distance))
            for name, (center, distance) in cases.items()}


def gate_fingerprints():
    from agent_test_support import load_agent_methods
    switch_sets = {
        'default': {},
        'confirmation_off': {'ENABLE_INSTANCE_CONFIRMATION': 'false'},
        'post_check': {'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'true'},
        'preserve': {'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'true'},
        'collect_only': {'INSTANCE_COLLECT_ONLY': 'true'},
        'collect_and_post': {'INSTANCE_COLLECT_ONLY': 'true',
                             'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'true'},
    }
    methods = ('_post_check_enabled', 'preserve_evidence_on_recovery_enabled',
               '_evidence_tracking_enabled', 'collect_only_enabled', 'defer_goal_lock')
    agent_class = load_agent_methods('get_goal_detection_threshold')
    out = {}
    for dataset in ('instancenav', 'hssd', 'hm3d_ovon', 'mp3d'):
        for label, switches in switch_sets.items():
            with env(**switches):
                manager = detection_manager(dataset=dataset)
                row = {}
                for method in methods:
                    try:
                        row[method] = bool(getattr(manager, method)())
                    except ValueError:
                        row[method] = 'raise:ValueError'
                agent = SimpleNamespace(dataset=dataset, obj_goal='chair', obj_goal_sg='chair',
                                        scenegraph=SimpleNamespace(threshold_list={'chair': 5},
                                                                   cfg=SimpleNamespace(obj_min_detections=5)))
                row['goal_threshold'] = agent_class.get_goal_detection_threshold(agent)
                out[f'{dataset}/{label}'] = row
    return out


def category_fingerprints():
    from configs import dataset_categories as dc
    from utils import utils_glip
    categories = ('chair', 'potted_plant', 'plant', 'tv', 'tv_monitor', 'couch', 'sofa',
                  'toilet', 'bed', 'cabinet')

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError:
            return 'raise:ValueError'

    out = {}
    for dataset in ('instancenav', 'hssd', 'hm3d_ovon', 'hm3d'):
        row = {'categories': call(dc.get_categories, dataset),
               'categories_cabinet': call(dc.get_categories, dataset, goal_category='cabinet'),
               'detection': call(dc.get_detection_categories, dataset, goal_category='cabinet'),
               'small': call(dc.get_small_objects, dataset),
               'count': call(dc.get_category_count, dataset, goal_category='cabinet'),
               'episode_goals': call(dc.uses_episode_goal_categories, dataset)}
        for category in categories:
            row[f'threshold/{category}'] = call(dc.get_threshold, category, dataset)
            row[f'to_unified/{category}'] = call(dc.dataset_to_unified_name, category, dataset)
            row[f'to_dataset/{category}'] = call(dc.unified_to_dataset_name, category, dataset)
        out[dataset] = row
    for dataset, goal in (('instancenav', None), ('hssd', None), ('hm3d_ovon', 'cabinet')):
        utils_glip.init_categories(dataset, goal_category=goal)
        out[f'glip/{dataset}'] = {
            'origin': list(utils_glip.categories_21_origin),
            'all': list(utils_glip.categories_21),
            'captions': utils_glip.object_captions,
            'projection': sorted((str(k), str(v)) for k, v in utils_glip.projection.items()),
        }
    return out


def node_space_fingerprints():
    from graph.meta_func.io_func import IOFunc
    out = {}
    for dataset, goal in (('instancenav', 'chair'), ('hm3d_ovon', 'cabinet'),
                          ('hm3d_ovon', 'chair'), (None, 'cabinet'), ('hssd', 'cabinet')):
        graph = SimpleNamespace(dataset=dataset, node_space='chair. table. bed.',
                                threshold_list={'chair': 4}, cfg=SimpleNamespace(obj_min_detections=1))
        IOFunc.set_obj_goal(graph, goal, goal)
        out[f'{dataset}/{goal}'] = [graph.node_space, graph.cfg.obj_min_detections]
    return out


def map_depth_key_fingerprint():
    """MapManager.update_map / update_free_map feed observations['depth'] to the modules."""
    from utils.map_manager import MapManager
    seen = []

    def module(depth, pose, last, *args):
        seen.append([list(depth.shape), float(depth.sum())])
        return last

    agent = SimpleNamespace(device=torch.device('cpu'), full_pose=torch.zeros(3),
                            full_map=torch.zeros(1, 1, 40, 40), fbe_free_map=torch.zeros(40, 40),
                            sem_map_module=module, free_map_module=module)
    manager = MapManager.__new__(MapManager)
    manager.agent, manager.device, manager.map_size_cm = agent, torch.device('cpu'), 200
    manager.map_resolution, manager.prev_agent_map_pos = 5, None
    obs = {'depth': np.full((4, 5, 1), 2.0, dtype=np.float32),
           'gps': np.array([.1, .2], dtype=np.float32), 'compass': np.array([.3], dtype=np.float32)}
    manager.update_map(obs)
    manager.update_free_map(obs)
    return seen


def all_fingerprints():
    return {'mapping': mapping_fingerprints(), 'stop_rule': stop_rule_fingerprints(),
            'fallback_depth': fallback_depth_fingerprints(), 'gates': gate_fingerprints(),
            'categories': category_fingerprints(), 'node_space': node_space_fingerprints(),
            'map_depth_key': map_depth_key_fingerprint()}
