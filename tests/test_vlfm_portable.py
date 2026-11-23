"""Portable VLFM layout and isolation. No checkpoints, model servers, or GPU needed."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nav_bases import vlfm_runtime as runtime
from nav_bases.vlfm_depth import fill_normalized_depth


ROOT = runtime.ROOT


def test_copy_layout_and_missing_assets(tmp_path, monkeypatch):
    monkeypatch.setenv('VLFM_ASSET_DIR', str(tmp_path))
    monkeypatch.delenv('VLFM_POINTNAV_WEIGHTS', raising=False)
    (tmp_path / runtime.POINTNAV).write_bytes(b'weights')
    assert runtime.check_assets(['navigation']) == [tmp_path / runtime.POINTNAV]
    (tmp_path / 'mobile_sam.pt').touch()  # a zero-byte failed transfer isn't a model
    (tmp_path / 'yolov7-e6e.pt').symlink_to(tmp_path / 'gone')
    with pytest.raises(FileNotFoundError) as err:
        runtime.check_assets()
    for asset in ('bert-base-uncased/vocab.txt', 'mobile_sam.pt', 'blip2_pretrained.pth',
                  'eva_vit_g.pth', 'groundingdino_swint_ogc.pth', 'yolov7-e6e.pt'):
        assert str(tmp_path / asset) in str(err.value)
    assert runtime.POINTNAV not in str(err.value)


def test_copy_manifest_covers_every_required_asset():
    entries = {name: digest for digest, name in (line.split() for line in
               (runtime.SOURCE / 'weights.sha256').read_text().splitlines())}
    assert {n for names in runtime.ASSETS.values() for n in names} <= entries.keys()
    assert entries['yolov7-e6e.pt'] == runtime.YOLO_SHA256


def test_bundled_policy_loads_without_a_sibling_checkout(tmp_path):
    pytest.importorskip('open3d')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(str(ROOT / p) for p in ('habitat-lab', '.', 'GLIP')),
           'VLFM_ROOT': '/does/not/exist', 'VLFM_DEPS': '/does/not/exist', 'VLFM_ASSET_DIR': str(tmp_path)}
    code = '''
import sys
from pathlib import Path
from nav_bases import vlfm_agent, vlfm_runtime
assert str(vlfm_runtime.SOURCE) not in sys.path  # importing the adapter alone is inert
cls = vlfm_agent.build_policy_class()
from vlfm.policy.itm_policy import ITMPolicyV2
assert issubclass(cls, ITMPolicyV2)
import vlfm, frontier_exploration, utils, habitat
assert Path(vlfm.__file__).resolve().is_relative_to(vlfm_runtime.SOURCE)
assert Path(frontier_exploration.__file__).resolve().is_relative_to(vlfm_runtime.SOURCE)
assert Path(utils.__file__).resolve().parent == vlfm_runtime.ROOT / 'utils'
assert Path(habitat.__file__).resolve().is_relative_to(vlfm_runtime.ROOT / 'habitat-lab')
assert 'models.yolo' not in sys.modules and 'lavis_itm' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path, check=True)


def test_depth_only_fills_holes_and_keeps_observed_samples():
    depth = np.full((48, 48), .4, dtype=np.float32)
    depth[20:25, 20:25] = 0
    depth[5, 8] = .05
    filled = fill_normalized_depth(depth)
    assert np.allclose(filled[20:25, 20:25], .4, atol=2e-5)
    np.testing.assert_array_equal(filled[depth != 0], depth[depth != 0])
    np.testing.assert_array_equal(fill_normalized_depth(np.zeros_like(depth)), 0)
