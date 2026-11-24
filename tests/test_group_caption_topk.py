"""Exact top-k vs the historical product → stable descending sort → slice."""
from functools import partial
from itertools import product
import random
import struct

import numpy as np
import pytest

from graph.caption_combinations import scored_captions
from graph.edge_class import Edge
from graph.enumerated_groups import EnumeratedGroupNodes
from graph.meta_func.core_func import CoreFunc
from graph.node_class import GroupNode, ObjectNode, RoomNode


def oracle(captions, probs, threshold, k):
    result = []
    for labels, values in zip(product(*captions), product(*probs)):
        p = 1.0
        for value in values:
            p *= value
        if not p < threshold:
            result.append((labels, p))
    return sorted(result, key=lambda item: item[1], reverse=True)[:k]


def bits(p):
    return struct.pack('>d', p)


def check(probs, *, k=3, threshold=0.0):
    captions = [[f'{i}:{j}' for j in range(len(axis))] for i, axis in enumerate(probs)]
    expected = oracle(captions, probs, threshold, k)
    found = [(c, p) for c, p in scored_captions(captions, probs, k) if not p < threshold]
    found = sorted(found, key=lambda item: item[1], reverse=True)[:k]
    assert [(c, bits(p)) for c, p in found] == [(c, bits(p)) for c, p in expected]


@pytest.mark.parametrize('probs', [
    [[.5, .5]] * 5,                           # stable ties across whole product
    [[.5, .25], [1., .5]],                    # equal scores from different factors
    [[.75, .25], [.5, 0., -0.]],              # zero ties, including sign bit
    [[.4, .6], [0., .75, .25]],               # unsorted, original order must survive
    [[1e-200, 1e-250], [1e-200, 1e-250]],      # underflow to tied zeros
    [[.5, float(np.nextafter(.5, 0))], [5e-324]],  # rounding in the fast path
    [[np.float32(.5), .25], [np.float16(.8), .75]],  # preserve mixed scalar arithmetic
    [[1.], [], [.5]], [], [[1.]],              # empty/singleton axes
    [[float('nan'), 0.5], [1., .5]],           # full-sort compatibility path
])
@pytest.mark.parametrize('k', [None, -1, 0, 1, 3, 20])
def test_exact_scores_and_stable_order(probs, k):
    for threshold in (0.0, 0.25, 1.):
        check(probs, k=k, threshold=threshold)


def test_fixed_seed_distributions_match_exhaustive():
    rng = random.Random(171)
    for _ in range(400):
        probs = [[rng.choice([0., 1., .125, .5, rng.random()]) for _ in range(rng.randint(1, 4))]
                 for _ in range(rng.randint(1, 6))]
        if rng.choice([True, False]):
            for axis in probs:
                axis.sort(reverse=True)
        check(probs, k=rng.choice([1, 2, 3, 8]), threshold=rng.choice([0., 0.1, 0.25]))


def make_group(objects):
    group = GroupNode()
    for i, obj in enumerate(objects):
        node = ObjectNode()
        node.object = obj
        node.caption = f'node{i}'
        node.center = np.array([i, 2])
        group.nodes.append(node)
    if len(group.nodes) > 1:
        Edge(*group.nodes[:2]).set_relation('beside')
    if group.nodes:
        group.get_graph()
    return group


def signature(group):
    return (group.caption, tuple((n.caption, n.center.tolist(), n.object, n.edges) for n in group.nodes),
            bits(group.probability), tuple(group.center) if group.center is not None else None,
            group.center_node.caption if group.center_node is not None else None, group.edges)


@pytest.mark.parametrize('threshold', [0., .1, .25, .75])
def test_downstream_graphs_normalization_and_group_indices_unchanged(threshold):
    room = RoomNode('study')
    room.group_nodes = [make_group([None, {'captions': ['a', 'a', 'b']}, {'captions': []},
                                    {'captions_sorted': ['z', 'c'], 'caption_probs': {'c': .5, 'z': .5}}]),
                        make_group([{'captions_sorted': ['b', 'a'], 'caption_probs': {'a': .75, 'b': .25}}]),
                        make_group([])]
    core = CoreFunc()
    old, new = EnumeratedGroupNodes(), EnumeratedGroupNodes()
    old.add_from_rooms([room], core.enumerate_group_node, top_k_per_group=3, min_prob_threshold=threshold)
    new.add_from_rooms([room], partial(core.enumerate_group_node, top_k=3), min_prob_threshold=threshold)
    def records(items):
        return [(x['room_node'], x['original_group'], x['group_idx'], x['enumeration_idx'],
                 x['caption_combo'], bits(x['probability']), signature(x['group_node'])) for x in items]
    assert records(new.all_enumerations) == records(old.all_enumerations)
    assert records(new.by_room['study']) == records(old.by_room['study'])
    assert {k: list(map(signature, v)) for k, v in new.by_group.items()} == {
        k: list(map(signature, v)) for k, v in old.by_group.items()}
    assert [n.caption for n in room.group_nodes[0].nodes] == ['node0', 'node1', 'node2', 'node3']


def test_large_tied_product_only_builds_three_graphs(monkeypatch):
    # 2**64 combinations. Guard the legacy iterator so a regression fails, not OOMs.
    import graph.caption_combinations as selection
    def forbidden_product(*axes):
        raise AssertionError('sorted top-k should not iterate the full Cartesian product')
    monkeypatch.setattr(selection, 'product', forbidden_product)
    original_copy = GroupNode.copy
    copies = []
    def count_copy(group):
        copies.append(1)
        assert len(copies) <= 3
        return original_copy(group)
    monkeypatch.setattr(GroupNode, 'copy', count_copy)
    group = make_group([{'captions': ['a', 'b']}] * 64)
    found = CoreFunc().enumerate_group_node(group, 0., top_k=3)
    assert [''.join(n.caption for n in g.nodes) for g in found] == ['a'*64, 'a'*63+'b', 'a'*62+'ba']
    assert len(copies) == 3
