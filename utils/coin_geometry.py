"""Camera and depth geometry for CoIN-Bench under Context-Nav's protocol.

Context-Nav's CoIN config (copied from the official CoIN config) renders RGB at
512x512 with HFOV 42 and depth at 512x512 with HFOV 79 (inherited from
habitat-lab 0.2.4), both at 1.31 m on a 1.41 m agent, depth clipped to 0-5 m.
PSG-Nav assumes RGB and depth share intrinsics and that camera height equals
agent height, which holds for InstanceNav (640x480, 79 deg, 0.88 m) but not here.

CoIN therefore gets:
  * depth_wide: the sanitized 79-degree depth, used for the obstacle and
    free-space maps (Context-Nav also builds its obstacle map from it);
  * depth: depth resampled into the RGB frame, used wherever an RGB pixel,
    box or mask is paired with depth, together with the RGB intrinsics;
  * projection from the real camera height (1.31 m), not the agent height.

InstanceNav never builds a CoinGeometry, so none of this runs on its path.
"""
import math

import numpy as np

# Same marker the InstanceNav path writes into invalid depth (act(): 0.5 -> 100).
# It lies above every configured MAX_DEPTH, so the mapper drops it, and the
# scene graph drops depth >= 60 m.
INVALID_DEPTH_M = 100.0
# Habitat clips depth to MAX_DEPTH exactly; treat values at the clip as no return.
MAX_DEPTH_TOLERANCE_M = 1e-4
STOP_DISTANCE_METRICS = ('fmm', 'euclidean')


def _focal(width, hfov_deg):
    """Pinhole focal length in pixels, matching depth_utils.get_camera_matrix."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))


def stop_distance_metric(config):
    """'fmm' (InstanceNav/CoIN default) or 'euclidean' (legacy replay comparison)."""
    metric = str(getattr(config.TASK, 'STOP_DISTANCE_METRIC', 'fmm')).lower()
    if metric not in STOP_DISTANCE_METRICS:
        raise ValueError(f'TASK.STOP_DISTANCE_METRIC must be one of {STOP_DISTANCE_METRICS}, got {metric!r}')
    return metric


def euclidean_goal_distance_m(state, goal_map, map_resolution_cm):
    """Straight-line distance (m) from planner cell `state` to the nearest goal cell.

    Context-Nav stops when the 2D Euclidean distance to the closest point of the
    target instance is below pointnav_stop_radius; this is the same test on the
    goal cells PSG-Nav hands to its planner. Returns inf for an empty goal map.
    """
    cells = np.argwhere(np.asarray(goal_map) != 0)
    if cells.size == 0:
        return math.inf
    deltas = cells.astype(np.float64) - np.asarray(state, dtype=np.float64)[None, :2]
    return float(np.sqrt((deltas ** 2).sum(axis=1)).min()) * float(map_resolution_cm) / 100.0


class CoinGeometry:
    """Sensor geometry for one CoIN run, built once from the Habitat config."""

    def __init__(self, rgb_hw, depth_hw, rgb_hfov_deg, depth_hfov_deg, min_depth_m, max_depth_m, camera_height_m):
        self.rgb_height, self.rgb_width = (int(v) for v in rgb_hw)
        self.depth_height, self.depth_width = (int(v) for v in depth_hw)
        self.rgb_hfov_deg = float(rgb_hfov_deg)
        self.depth_hfov_deg = float(depth_hfov_deg)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.camera_height_m = float(camera_height_m)
        self.camera_height_cm = self.camera_height_m * 100.0
        if self.depth_hfov_deg < self.rgb_hfov_deg:
            raise ValueError('CoIN geometry needs the depth view to cover the RGB view '
                             f'(depth HFOV {self.depth_hfov_deg} < RGB HFOV {self.rgb_hfov_deg})')
        if not 0.0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError(f'bad depth range [{self.min_depth_m}, {self.max_depth_m}] m')
        self.rgb_focal = _focal(self.rgb_width, self.rgb_hfov_deg)
        self.depth_focal = _focal(self.depth_width, self.depth_hfov_deg)
        # Both sensors share position and orientation, so the ray through RGB pixel
        # (v, u) passes through depth pixel c_d + (u - c_r) * f_d / f_r (same for
        # rows; Habitat pixels are square), and the planar depth z along the shared
        # optical axis is identical in both frames. Nearest neighbour keeps depth
        # edges sharp instead of blending foreground and background.
        scale = self.depth_focal / self.rgb_focal
        cols = (self.depth_width - 1) / 2.0 + (np.arange(self.rgb_width) - (self.rgb_width - 1) / 2.0) * scale
        rows = (self.depth_height - 1) / 2.0 + (np.arange(self.rgb_height) - (self.rgb_height - 1) / 2.0) * scale
        self.aligned_cols = np.clip(np.rint(cols), 0, self.depth_width - 1).astype(np.int64)
        self.aligned_rows = np.clip(np.rint(rows), 0, self.depth_height - 1).astype(np.int64)

    @classmethod
    def from_config(cls, config):
        """Build from a Habitat config; reject layouts the resampling cannot handle."""
        sim = config.SIMULATOR
        rgb, depth = sim.RGB_SENSOR, sim.DEPTH_SENSOR
        for key in ('POSITION', 'ORIENTATION'):
            rgb_value = [float(v) for v in getattr(rgb, key, [0.0, 0.0, 0.0])]
            depth_value = [float(v) for v in getattr(depth, key, [0.0, 0.0, 0.0])]
            if not np.allclose(rgb_value, depth_value):
                raise ValueError(f'CoIN geometry needs co-located sensors: RGB {key} {rgb_value} != depth {depth_value}')
        if (int(rgb.HEIGHT), int(rgb.WIDTH)) != (int(depth.HEIGHT), int(depth.WIDTH)):
            # Room masks are drawn on RGB pixels and projected with RGB-aligned depth,
            # so both images must have one pixel grid.
            raise ValueError('CoIN geometry needs equal RGB and depth image sizes, got '
                             f'{rgb.HEIGHT}x{rgb.WIDTH} vs {depth.HEIGHT}x{depth.WIDTH}')
        if bool(getattr(depth, 'NORMALIZE_DEPTH', False)):
            raise ValueError('CoIN geometry needs metric depth: set DEPTH_SENSOR.NORMALIZE_DEPTH False')
        stop_distance_metric(config)
        return cls(
            rgb_hw=(rgb.HEIGHT, rgb.WIDTH),
            depth_hw=(depth.HEIGHT, depth.WIDTH),
            rgb_hfov_deg=rgb.HFOV,
            depth_hfov_deg=depth.HFOV,
            min_depth_m=depth.MIN_DEPTH,
            max_depth_m=depth.MAX_DEPTH,
            camera_height_m=float(depth.POSITION[1]),
        )

    def sanitize(self, depth):
        """Mark no-return depth with INVALID_DEPTH_M; returns a new float32 array.

        Habitat clips depth to [MIN_DEPTH, MAX_DEPTH]: holes read MIN_DEPTH (0 here)
        and anything farther reads exactly MAX_DEPTH. The mapper would accept both as
        hits (a 0 lands on the camera itself; a tilted clip lands inside the 5 m
        grid), so both become the invalid marker, as 0.5 does on InstanceNav.
        """
        depth = np.array(depth, dtype=np.float32, copy=True)
        invalid = ~np.isfinite(depth)
        invalid |= depth <= self.min_depth_m
        invalid |= depth >= self.max_depth_m - MAX_DEPTH_TOLERANCE_M
        depth[invalid] = INVALID_DEPTH_M
        return depth

    def align_to_rgb(self, depth_wide):
        """Resample sanitized wide depth onto the RGB pixel grid (H, W, 1)."""
        if depth_wide.shape[:2] != (self.depth_height, self.depth_width):
            raise ValueError(f'depth shape {depth_wide.shape[:2]} != configured '
                             f'{(self.depth_height, self.depth_width)}')
        return np.ascontiguousarray(depth_wide[self.aligned_rows[:, None], self.aligned_cols[None, :]])

    def prepare_observations(self, observations):
        """Replace observations['depth'] in place, as act() does for InstanceNav.

        After this, 'depth' is RGB-aligned (paired with RGB boxes, masks and the
        RGB camera matrix) and 'depth_wide' is the full 79-degree depth for the
        obstacle and free-space maps. A dict that already has 'depth_wide' was
        prepared before and is left alone.
        """
        if 'depth_wide' in observations:
            return observations
        depth_wide = self.sanitize(observations['depth'])
        observations['depth_wide'] = depth_wide
        observations['depth'] = self.align_to_rgb(depth_wide)
        return observations

    def pixel_direction(self, x):
        """Horizontal angle (deg, right positive) of RGB column x, exact pinhole.

        InstanceNav keeps its linear (x - 320) * 79 / 640; at 42 degrees the
        linear and exact forms differ by under 0.4 degrees, but this one is the
        sensor's actual ray direction.
        """
        return math.degrees(math.atan((float(x) - (self.rgb_width - 1) / 2.0) / self.rgb_focal))

    def rgb_camera_matrix(self):
        """RGB intrinsics in the Namespace form SceneGraph and Semantic_Mapping use."""
        from utils.utils_fmm.depth_utils import get_camera_matrix
        return get_camera_matrix(self.rgb_width, self.rgb_height, self.rgb_hfov_deg)

    def describe(self):
        return {
            'rgb_hw': [self.rgb_height, self.rgb_width], 'rgb_hfov_deg': self.rgb_hfov_deg,
            'depth_hw': [self.depth_height, self.depth_width], 'depth_hfov_deg': self.depth_hfov_deg,
            'rgb_focal_px': round(self.rgb_focal, 3), 'depth_focal_px': round(self.depth_focal, 3),
            'depth_range_m': [self.min_depth_m, self.max_depth_m],
            'camera_height_m': self.camera_height_m,
            'aligned_cols': [int(self.aligned_cols[0]), int(self.aligned_cols[-1])],
            'aligned_rows': [int(self.aligned_rows[0]), int(self.aligned_rows[-1])],
        }
