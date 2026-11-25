"""Optional real-data audits, not part of default pytest collection.

From the project root in the navigation environment:
    python -m pytest tests/audit_instancenav_sources.py

These checks need the original PSL/IIN downloads, not just the prepared 100 tasks.
"""
import json
import zipfile
from pathlib import Path

import pytest

from tools.prepare_instancenav import prepare


def require_inputs(*paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.fail('Missing inputs for the requested InstanceNav source audit: ' + ', '.join(missing),
                    pytrace=False)


def test_real_dataset_subset_reproducible_and_instance_specific(tmp_path):
    attributes = Path('data/datasets/instancenav/val/val_text.json.gz')
    content = Path('data/datasets/instance_imagenav_hm3d_v3/val/content')
    require_inputs(attributes, content)
    output = tmp_path / 'a.json.gz'
    a = prepare(attributes, content, output)
    b = prepare(attributes, content, tmp_path / 'b.json.gz')
    assert (a['source_count'], a['selected_count']) == (1000, 200)
    assert a['episodes'] == b['episodes']
    assert a['prepared_sha256'] == b['prepared_sha256']

    import habitat
    from utils.instancenav_dataset import InstanceNavDataset
    config = habitat.get_config('configs/instancenav_val20_qwen.yaml',
                                opts=['DATASET.DATA_PATH', str(output)])
    assert config.ENVIRONMENT.MAX_EPISODE_STEPS == 1000
    assert config.TASK.SUCCESS.SUCCESS_DISTANCE == 1.0
    dataset = InstanceNavDataset(config.DATASET)
    assert len(dataset.episodes) == 200
    for ep, expected in zip(dataset.episodes, a['episodes']):
        assert len(ep.goals) == 1
        assert str(ep.goals[0].object_id) == str(expected['goal_object_id'])
        assert ep.goal_attributes


def test_unseen_subset_has_no_training_scene_overlap():
    archive = Path('data/downloads/instance_imagenav_hm3d_v3.zip')
    manifest = Path('data/datasets/instancenav/val20/val20.manifest.json')
    require_inputs(archive, manifest)
    selected = {Path(s).name.split('.')[0] for s in json.loads(manifest.read_text())['scenes']}
    with zipfile.ZipFile(archive) as z:
        train = {Path(n).name.removesuffix('.json.gz') for n in z.namelist()
                 if '/train/content/' in n and n.endswith('.json.gz')}
        val = {Path(n).name.removesuffix('.json.gz') for n in z.namelist()
               if '/val/content/' in n and n.endswith('.json.gz')}
    assert len(train) == 145 and len(val) == 36
    assert selected <= val
    assert selected.isdisjoint(train)
