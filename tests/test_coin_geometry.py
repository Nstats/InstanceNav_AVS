"""CoIN camera/depth geometry: alignment, sanitizing, projection height, intrinsics."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.coin_geometry import INVALID_DEPTH_M, CoinGeometry, euclidean_goal_distance_m, stop_distance_metric

COIN_CONFIG = 'configs/coin_contextnav.yaml'


@pytest.fixture(scope='module')
def coin_config():
    import habitat
    return habitat.get_config(COIN_CONFIG)


@pytest.fixture(scope='module')
def geometry(coin_config):
    return CoinGeometry.from_config(coin_config)


def test_constants_follow_context_nav(geometry):
    assert (geometry.rgb_height, geometry.rgb_width, geometry.rgb_hfov_deg) == (512, 512, 42.0)
    assert (geometry.depth_height, geometry.depth_width, geometry.depth_hfov_deg) == (512, 512, 79.0)
    assert (geometry.min_depth_m, geometry.max_depth_m) == (0.0, 5.0)
    assert geometry.camera_height_cm == pytest.approx(131.0)


def test_aligned_index_is_the_nearest_depth_pixel_on_the_same_ray(geometry):
    # Shared optical centre: tan(angle) = (u - c_r) / f_r = (u_d - c_d) / f_d.
    u = np.arange(geometry.rgb_width)
    exact = 255.5 + (u - 255.5) * geometry.depth_focal / geometry.rgb_focal
    assert np.all(np.abs(geometry.aligned_cols - exact) <= 0.5)
    assert np.array_equal(geometry.aligned_rows, geometry.aligned_cols)  # square pixels, square images
    assert (geometry.aligned_cols[0], geometry.aligned_cols[-1]) == (137, 374)
    assert np.all(np.diff(geometry.aligned_cols) >= 0)


def test_sanitize_marks_holes_clip_and_nan(geometry):
    depth = np.array([[0.0, 0.3, 4.99, 5.0, np.nan, np.inf]], dtype=np.float32)[..., None]
    out = geometry.sanitize(depth)[0, :, 0]
    assert out.tolist() == [INVALID_DEPTH_M, pytest.approx(0.3), pytest.approx(4.99), INVALID_DEPTH_M,
                            INVALID_DEPTH_M, INVALID_DEPTH_M]
    assert depth[0, 0, 0] == 0.0  # input untouched


def test_prepare_observations_replaces_depth_once(geometry):
    wide = np.random.default_rng(0).uniform(0.2, 4.8, (512, 512, 1)).astype(np.float32)
    obs = {'depth': wide.copy(), 'rgb': np.zeros((512, 512, 3), np.uint8)}
    geometry.prepare_observations(obs)
    assert obs['depth_wide'].shape == obs['depth'].shape == (512, 512, 1)
    assert obs['depth'].dtype == np.float32 and obs['depth'].flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(obs['depth_wide'], wide)
    np.testing.assert_array_equal(obs['depth'][0, 0], wide[137, 137])
    np.testing.assert_array_equal(obs['depth'][511, 511], wide[374, 374])
    aligned = obs['depth']
    geometry.prepare_observations(obs)  # second call must not re-crop
    assert obs['depth'] is aligned


def test_rgb_intrinsics_and_pixel_direction(geometry):
    from utils.utils_fmm.depth_utils import get_camera_matrix
    cm = geometry.rgb_camera_matrix()
    ref = get_camera_matrix(512, 512, 42)
    assert (cm.xc, cm.zc, cm.f) == (ref.xc, ref.zc, ref.f)
    assert geometry.pixel_direction(255.5) == pytest.approx(0.0)
    assert geometry.pixel_direction(511) == pytest.approx(-geometry.pixel_direction(0))
    assert geometry.pixel_direction(511) == pytest.approx(math.degrees(math.atan(255.5 / cm.f)))
    assert geometry.pixel_direction(torch.tensor(400)) > 0  # right of centre is positive, tensors accepted


def _sensor_config(**overrides):
    sensor = dict(HEIGHT=512, WIDTH=512, HFOV=42, POSITION=[0, 1.31, 0], ORIENTATION=[0, 0, 0])
    rgb = SimpleNamespace(**sensor)
    depth = SimpleNamespace(**{**sensor, 'HFOV': 79, 'MIN_DEPTH': 0.0, 'MAX_DEPTH': 5.0, 'NORMALIZE_DEPTH': False})
    for key, value in overrides.items():
        target, field = key.split('__')
        setattr(rgb if target == 'rgb' else depth, field, value)
    return SimpleNamespace(SIMULATOR=SimpleNamespace(RGB_SENSOR=rgb, DEPTH_SENSOR=depth),
                           TASK=SimpleNamespace(STOP_DISTANCE_METRIC='euclidean'))


@pytest.mark.parametrize('overrides, message', [
    ({'depth__POSITION': [0, 1.0, 0]}, 'co-located'),
    ({'rgb__ORIENTATION': [0.1, 0, 0]}, 'co-located'),
    ({'depth__WIDTH': 640}, 'equal RGB and depth image sizes'),
    ({'depth__HFOV': 30}, 'cover the RGB view'),
    ({'depth__NORMALIZE_DEPTH': True}, 'metric depth'),
])
def test_from_config_rejects_unsupported_layouts(overrides, message):
    CoinGeometry.from_config(_sensor_config())
    with pytest.raises(ValueError, match=message):
        CoinGeometry.from_config(_sensor_config(**overrides))


def _mapping_agent(config):
    return SimpleNamespace(device=torch.device('cpu'), config=config, map_size_cm=4000)


def test_semantic_mapping_overrides(coin_config, geometry):
    from utils.utils_fmm.mapping import Semantic_Mapping
    agent = _mapping_agent(coin_config)
    default = Semantic_Mapping(agent)
    assert default.agent_height == pytest.approx(141.0) and default.fov == 79
    sem = Semantic_Mapping(agent, camera_height_cm=geometry.camera_height_cm, fov=geometry.depth_hfov_deg)
    assert sem.agent_height == pytest.approx(131.0)
    assert (sem.min_z_consider, sem.max_z_consider) == (20, pytest.approx(182.0))  # camera height + 51
    free = Semantic_Mapping(agent, max_height=10, min_height=-150, camera_height_cm=131.0, fov=79)
    assert (free.min_z_consider, free.max_z_consider) == (-150, 10)
    room = Semantic_Mapping(agent, max_height=200, min_height=-10, num_cats=9, camera_height_cm=131.0, fov=42)
    assert room.camera_matrix.f == pytest.approx(geometry.rgb_focal)
    assert sem.camera_matrix.f == pytest.approx(geometry.depth_focal)
    assert (sem.min_depth_cm, sem.max_depth_cm) == (0.0, 500.0)


def _plane_depth(height_above_floor_m, camera_height_m=1.31, size=512, hfov=79.0):
    """Planar depth (m) of a level, infinite horizontal plane seen by a level camera."""
    f = (size / 2.0) / math.tan(math.radians(hfov / 2.0))
    below_centre = np.arange(size, dtype=np.float32) - (size - 1) / 2.0  # >0 looks down
    with np.errstate(divide='ignore'):
        rows = (camera_height_m - height_above_floor_m) * f / below_centre
    rows = np.where((below_centre > 0) & (rows < 4.9), rows, INVALID_DEPTH_M)
    return np.repeat(rows[:, None], size, axis=1).astype(np.float32)


def test_low_surface_is_not_an_obstacle_only_at_the_true_camera_height(coin_config):
    """A 15 cm high surface lies under the 20 cm obstacle band at the true 1.31 m
    camera height; projecting from the 1.41 m agent height lifts it to 25 cm."""
    from utils.utils_fmm.mapping import Semantic_Mapping
    agent = _mapping_agent(coin_config)
    depth = torch.from_numpy(_plane_depth(0.15))
    pose = torch.tensor([20.0, 20.0, 0.0])
    counts = {}
    for label, height in (('camera', 131.0), ('agent', None)):
        module = Semantic_Mapping(agent, camera_height_cm=height, fov=79)
        module.set_view_angles(0)
        counts[label] = int(module(depth, pose, torch.zeros(1, 1, 800, 800)).sum())
    assert counts['camera'] == 0
    assert counts['agent'] > 100


def test_floor_is_free_space_only_at_the_true_camera_height(coin_config):
    """Projected from 1.41 m the floor lands at +10 cm, outside the free band [-150, 10),
    so the free-space map (frontiers) would stay empty."""
    from utils.utils_fmm.mapping import Semantic_Mapping
    agent = _mapping_agent(coin_config)
    depth = torch.from_numpy(_plane_depth(0.0))
    pose = torch.tensor([20.0, 20.0, 0.0])
    counts = {}
    for label, height in (('camera', 131.0), ('agent', None)):
        module = Semantic_Mapping(agent, max_height=10, min_height=-150, camera_height_cm=height, fov=79)
        module.set_view_angles(0)
        counts[label] = int(module(depth, pose, torch.zeros(1, 1, 800, 800)).sum())
    assert counts['camera'] > 1000
    assert counts['agent'] == 0


def test_euclidean_goal_distance_and_metric():
    goal = np.zeros((40, 40))
    goal[20, 34] = goal[30, 20] = 1
    assert euclidean_goal_distance_m((20, 20), goal, 5) == pytest.approx(0.5)  # 10 cells to (30, 20)
    assert euclidean_goal_distance_m((20, 20), np.zeros((4, 4)), 5) == math.inf
    assert stop_distance_metric(SimpleNamespace(TASK=SimpleNamespace())) == 'fmm'
    assert stop_distance_metric(SimpleNamespace(TASK=SimpleNamespace(STOP_DISTANCE_METRIC='Euclidean'))) == 'euclidean'
    with pytest.raises(ValueError):
        stop_distance_metric(SimpleNamespace(TASK=SimpleNamespace(STOP_DISTANCE_METRIC='geodesic')))
