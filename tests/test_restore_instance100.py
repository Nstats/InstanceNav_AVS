import gzip
import json

import pytest

from tools.prepare_instancenav import prepare
from tools.restore_instance100 import restore, verify


def fixture_sources(tmp_path):
    attrs = tmp_path / 'attributes.json.gz'
    content = tmp_path / 'content'
    content.mkdir()
    episodes = [dict(episode_id=str(i), scene_id='scene.basis.glb',
                     goal_object_id=str(i), object_category='chair') for i in range(6)]
    with gzip.open(attrs, 'wt') as f:
        json.dump({'attribute_data': {f'scene_{i}': {'color': 'red'} for i in range(6)}}, f)
    with gzip.open(content / 'scene.json.gz', 'wt') as f:
        json.dump({'episodes': episodes, 'goals': {
            f'scene_{i}': {'view_points': [{'agent_state': {}}]} for i in range(6)}}, f)
    frozen = tmp_path / 'frozen.json.gz'
    manifest = prepare(attrs, content, frozen, fraction=0.5)
    order = tmp_path / 'order.json'
    order.write_text(json.dumps([dict(index=i, **ep) for i, ep in enumerate(manifest['episodes'])]))
    return attrs, content, frozen, tmp_path / 'frozen.manifest.json', order


def test_restore_reproduces_frozen_bytes_and_manifest(tmp_path):
    attrs, content, frozen, manifest, order = fixture_sources(tmp_path)
    output = tmp_path / 'destination/subset.json.gz'
    restore(attrs, content, output, manifest, order)
    assert output.read_bytes() == frozen.read_bytes()
    assert output.with_name('subset.manifest.json').read_bytes() == manifest.read_bytes()
    assert verify(output, manifest, order)['selected_count'] == 3


def test_changed_source_does_not_replace_existing_subset(tmp_path):
    attrs, content, frozen, manifest, order = fixture_sources(tmp_path)
    before = frozen.read_bytes()
    with gzip.open(attrs, 'rt') as f:
        values = json.load(f)
    for value in values['attribute_data'].values():
        value['color'] = 'blue'
    with gzip.open(attrs, 'wt') as f:
        json.dump(values, f)
    with pytest.raises(ValueError, match='SHA256'):
        restore(attrs, content, frozen, manifest, order)
    assert frozen.read_bytes() == before


def test_wrong_execution_index_is_rejected(tmp_path):
    _, _, frozen, manifest, order = fixture_sources(tmp_path)
    records = json.loads(order.read_text())
    records[0], records[1] = records[1], records[0]
    order.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='evaluation order'):
        verify(frozen, manifest, order)
