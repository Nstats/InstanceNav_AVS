"""InstanceNav behaviour must be identical before and after CoIN-Bench support.

The goldens were computed on commit 01af75a, before any CoIN change (see the
'_provenance' field). Each section covers one place the CoIN change touches.
"""
import json
from pathlib import Path

import pytest

import instancenav_characterization_support as support

# tests/goldens, not tests/data: .gitignore drops every data/ directory.
GOLDENS = json.loads((Path(__file__).parent / 'goldens' / 'instancenav_characterization.json').read_text())


def _normalise(value):
    return json.loads(json.dumps(value, sort_keys=True))


@pytest.mark.parametrize('section, compute', [
    ('mapping', support.mapping_fingerprints),
    ('stop_rule', support.stop_rule_fingerprints),
    ('fallback_depth', support.fallback_depth_fingerprints),
    ('gates', support.gate_fingerprints),
    ('categories', support.category_fingerprints),
    ('node_space', support.node_space_fingerprints),
    ('map_depth_key', support.map_depth_key_fingerprint),
])
def test_instancenav_path_unchanged(section, compute):
    assert _normalise(compute()) == GOLDENS[section]
