"""Runtime Habitat config: per-process atomic copy and post-load check.

Regression for the avs5_E3_r1 GPU 0 worker, which loaded an empty shared runtime yaml
(written concurrently by the GPU 1 worker) and ran the whole batch on Habitat defaults.
"""
import os
import threading
from pathlib import Path

import pytest

from utils.runtime_config import (
    assert_unchanged,
    build_runtime_config_file,
    config_mismatches,
    config_summary,
    inject_split,
    verify_loaded_config,
)

INSTANCE_CONFIG = 'configs/runtime/instancenav_unseen10_1000atomic_1m.yaml'


def _get_config(path):
    habitat = pytest.importorskip('habitat')
    return habitat.get_config(str(path))


def test_inject_split_replaces_only_dataset_split():
    text = 'SEED: 1\nTASK:\n  SPLIT: keep\nDATASET:\n  SPLIT: val\n  DATA_PATH: x\n'
    out = inject_split(text, 'val_unseen')
    assert '  SPLIT: keep' in out
    assert 'DATASET:\n  SPLIT: val_unseen\n  DATA_PATH: x' in out


def test_inject_split_requires_split_line():
    with pytest.raises(ValueError):
        inject_split('DATASET:\n  DATA_PATH: x\n', 'val')


def test_runtime_file_is_private_to_the_process(tmp_path):
    path, digest = build_runtime_config_file(INSTANCE_CONFIG, 'val', runtime_dir=tmp_path,
                                             remove_at_exit=False)
    assert Path(path).name == f'instancenav_unseen10_1000atomic_1m.val.{os.getpid()}.yaml'
    assert Path(path).read_text() == inject_split(Path(INSTANCE_CONFIG).read_text(), 'val')
    assert_unchanged(path, digest)
    assert not [p for p in tmp_path.iterdir() if p.suffix == '.tmp']


def test_concurrent_writers_never_expose_a_partial_file(tmp_path):
    """Rewriting the same per-process path while another thread reads it never yields
    an empty or truncated file (the old write_text truncated before writing)."""
    expected = inject_split(Path(INSTANCE_CONFIG).read_text(), 'val')
    path, _ = build_runtime_config_file(INSTANCE_CONFIG, 'val', runtime_dir=tmp_path,
                                        remove_at_exit=False)
    bad = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            content = Path(path).read_text()
            if content != expected:
                bad.append(len(content))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(300):
            build_runtime_config_file(INSTANCE_CONFIG, 'val', runtime_dir=tmp_path,
                                      remove_at_exit=False)
    finally:
        stop.set()
        thread.join()
    assert bad == []


def test_assert_unchanged_detects_rewrite(tmp_path):
    path, digest = build_runtime_config_file(INSTANCE_CONFIG, 'val', runtime_dir=tmp_path,
                                             remove_at_exit=False)
    Path(path).write_text('')
    with pytest.raises(RuntimeError, match='changed after it was written'):
        assert_unchanged(path, digest)


@pytest.mark.parametrize('base,split', [
    (INSTANCE_CONFIG, 'val'),
    ('configs/coin_contextnav.yaml', 'val_unseen'),
    ('configs/hm3d_ovon_unseen10_500atomic_1m.yaml', None),
])
def test_loaded_runtime_config_matches_base(tmp_path, base, split):
    import yaml
    split = split or yaml.safe_load(Path(base).read_text())['DATASET']['SPLIT']
    path, _ = build_runtime_config_file(base, split, runtime_dir=tmp_path, remove_at_exit=False)
    config = _get_config(path)
    assert config_mismatches(config, base, split) == []
    verify_loaded_config(config, base, split)


def test_empty_runtime_file_is_rejected(tmp_path):
    empty = tmp_path / 'empty.yaml'
    empty.write_text('')
    config = _get_config(empty)
    with pytest.raises(RuntimeError) as error:
        verify_loaded_config(config, INSTANCE_CONFIG, 'val')
    message = str(error.value)
    assert 'ENVIRONMENT.ATOMIC_ACTIONS' in message
    assert 'refusing to run' in message


def test_wrong_split_is_rejected(tmp_path):
    path, _ = build_runtime_config_file(INSTANCE_CONFIG, 'val', runtime_dir=tmp_path,
                                        remove_at_exit=False)
    config = _get_config(path)
    assert [m[0] for m in config_mismatches(config, INSTANCE_CONFIG, 'val_unseen')] == ['DATASET.SPLIT']


def test_empty_base_config_is_an_error(tmp_path):
    base = tmp_path / 'base.yaml'
    base.write_text('')
    assert config_mismatches({}, base, 'val') == [('<base config>', 'non-empty mapping', 'empty or invalid')]


def test_summary_records_policy_geometry():
    summary = config_summary(_get_config(INSTANCE_CONFIG))
    assert summary['ENVIRONMENT.ATOMIC_ACTIONS'] is True
    assert summary['SIMULATOR.DEPTH_SENSOR.HFOV'] == 79
    assert summary['SIMULATOR.DEPTH_SENSOR.POSITION'] == [0, 0.88, 0]
    assert summary['SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH'] is False
