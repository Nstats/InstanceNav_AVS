import json

import numpy as np
import pytest

from tools.offline_verifier_eval import (
    DiskCachedClient, _gain, build_checklist_prompt, checklist_schema, draw_mark, factor_key,
    parse_checklist, parse_rooms, parse_v1, rule_any, rule_match, rule_net, rule_or, rule_overall,
    rule_room, select_history,
)

FACTS = [{'id': 'a1', 'kind': 'attribute', 'text': 'black leather'},
         {'id': 'r1', 'kind': 'relation', 'text': 'next to a white desk'},
         {'id': 'r2', 'kind': 'relation', 'text': 'two monitors behind'}]


def _checklist(statuses, overall='unknown'):
    return json.dumps({'facts': [{'id': i, 'status': s, 'evidence': 'x'} for i, s in statuses],
                       'is_target': overall})


def test_factor_key_matches_the_factor_table_convention():
    assert factor_key({'key': ['yr17PDCnDDW.basis.glb', '903', '398']}) == 'yr17PDCnDDW_398__ep903'


def test_parse_checklist_requires_every_fact_in_order():
    ids = ['a1', 'r1', 'r2']
    ok = parse_checklist(_checklist([('a1', 'supported'), ('r1', 'contradicted'), ('r2', 'not_visible')], 'no'), ids)
    assert ok['facts'] == {'a1': 'supported', 'r1': 'contradicted', 'r2': 'not_visible'}
    assert ok['is_target'] == 'no'
    with pytest.raises(ValueError):
        parse_checklist(_checklist([('a1', 'supported'), ('r1', 'contradicted')]), ids)          # missing
    with pytest.raises(ValueError):
        parse_checklist(_checklist([('r1', 'supported'), ('a1', 'supported'), ('r2', 'supported')]), ids)  # order
    with pytest.raises(ValueError):
        parse_checklist(_checklist([('a1', 'maybe'), ('r1', 'supported'), ('r2', 'supported')]), ids)
    # Code fences and a singleton list wrapper are tolerated.
    wrapped = '```json\n[' + _checklist([('a1', 'supported'), ('r1', 'supported'), ('r2', 'supported')], 'yes') + ']\n```'
    assert parse_checklist(wrapped, ids)['is_target'] == 'yes'
    schema = checklist_schema(ids)
    assert schema['properties']['facts']['minItems'] == 3


def test_simple_parsers():
    assert parse_v1('{"match": "no"}') == 'no'
    with pytest.raises(ValueError):
        parse_v1('{"match": "no", "why": "x"}')
    assert parse_rooms('{"rooms": ["kitchen", "dining room", "kitchen"]}') == ['dining room', 'kitchen']
    with pytest.raises(ValueError):
        parse_rooms('{"rooms": ["attic"]}')


def test_relation_rules_use_only_relation_facts():
    res = parse_checklist(_checklist([('a1', 'contradicted'), ('r1', 'contradicted'), ('r2', 'supported')]),
                          ['a1', 'r1', 'r2'])
    assert rule_net(res, FACTS, 'relation') is False        # 1 contradicted vs 1 supported
    assert rule_any(res, FACTS, 'relation') is True
    assert rule_net(res, FACTS, 'attribute') is True
    res2 = parse_checklist(_checklist([('a1', 'supported'), ('r1', 'contradicted'), ('r2', 'not_visible')]),
                           ['a1', 'r1', 'r2'])
    assert rule_net(res2, FACTS, 'relation') is True
    # No relation facts in the task: abstain.
    assert rule_net(res2, FACTS[:1], 'relation') is None
    assert rule_net(None, FACTS, 'relation') is None


def test_match_overall_room_and_or_rules():
    assert rule_match('yes') is False and rule_match('unknown') is True and rule_match(None) is None
    assert rule_overall({'is_target': 'no'}) is True and rule_overall({'is_target': 'unknown'}) is False
    assert rule_room(['kitchen', 'dining room'], ['living room', 'dining room']) is False
    assert rule_room(['kitchen'], ['bedroom']) is True
    assert rule_room(['unknown'], ['bedroom']) is None
    assert rule_room(['kitchen'], ['other']) is None
    assert rule_or(None, None) is None and rule_or(False, None) is False and rule_or(None, True) is True


def test_select_history_agent_side_and_oracle():
    ep = {'boxes': [
        {'step': 10, 'est_gps': [0.5, 0.0], 'box': [0, 0, 10, 10], 'box_area_frac': 0.01, 'gt_instance': 7},
        {'step': 11, 'est_gps': [0.4, 0.0], 'box': [0, 0, 20, 20], 'box_area_frac': 0.02, 'gt_instance': 7},
        {'step': 20, 'est_gps': [3.0, 0.0], 'box': [0, 0, 5, 5], 'box_area_frac': 0.005, 'gt_instance': 7},
        {'step': 30, 'est_gps': [0.1, 0.1], 'box': [0, 0, 30, 30], 'box_area_frac': 0.05, 'gt_instance': 9},
        {'step': 50, 'est_gps': [0.0, 0.0], 'box': [0, 0, 50, 50], 'box_area_frac': 0.2, 'gt_instance': 7},
    ]}
    lock = {'step': 50, 'goal_gps': [0.0, 0.0]}
    frames = {10: 'f10', 11: 'f11', 20: 'f20', 30: 'f30', 50: 'f50'}
    agent = select_history(ep, lock, ep['boxes'][-1], frames)
    # Within 1 m and before the lock; steps 10 and 11 are closer than the minimum gap, keep the wider view.
    assert [h['step'] for h in agent] == [10, 30]
    oracle = select_history(ep, lock, ep['boxes'][-1], frames, oracle=True)
    assert [h['step'] for h in oracle] == [10, 20]
    assert all(h['step'] < lock['step'] for h in agent + oracle)


def test_gain_counts_recovered_wrong_stops_and_lost_correct_ones():
    rows = [{'run': 'a', 'group': 'adm', 'success': False, 'cls': 'ws_samecat_before', 'd': {'k': True}},
            {'run': 'a', 'group': 'adm', 'success': True, 'cls': 'success', 'd': {'k': True}},
            {'run': 'a', 'group': 'adm', 'success': False, 'cls': 'ws_other', 'd': {'k': False}}]
    assert _gain(rows, 'k') == pytest.approx(0.43 * 0.638 - 1.0)


def test_disk_cache_reuses_answers(tmp_path):
    class Fake:
        def __init__(self):
            self.n = 0

        def complete(self, prompt, image=None, **kw):
            self.n += 1
            return '{"match": "yes"}'

    from PIL import Image
    im = Image.new('RGB', (4, 4))
    fake = Fake()
    cc = DiskCachedClient(fake, tmp_path / 'a.jsonl')
    cc.local.tag = 'V1'
    assert cc.complete('p', im) == '{"match": "yes"}'
    assert cc.complete('p', im) == '{"match": "yes"}'
    assert fake.n == 1
    cc2 = DiskCachedClient(fake, tmp_path / 'a.jsonl')
    cc2.local.tag = 'V1'
    cc2.complete('p', im)
    assert fake.n == 1
    cc2.local.tag = 'V0#b'                                  # a different tag is a fresh request
    cc2.complete('p', im)
    assert fake.n == 2


def test_prompt_mentions_history_only_with_several_images():
    one = build_checklist_prompt('chair', {'intrinsic_attributes': 'x', 'extrinsic_attributes': 'y'}, FACTS, 1)
    three = build_checklist_prompt('chair', {'intrinsic_attributes': 'x', 'extrinsic_attributes': 'y'}, FACTS, 3)
    assert 'earlier views' not in one and 'Images 0 to 1 are earlier views' in three
    assert '"r2"' in one


def test_draw_mark_box_and_center():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    a = np.asarray(draw_mark(rgb, ('box', [5, 5, 20, 20])))
    b = np.asarray(draw_mark(rgb, ('center', [20, 20])))
    assert a[5, 10, 0] > 200 and b[20, 20, 0] > 200


def test_auc_and_best_threshold_bound():
    from tools.offline_verifier_eval import _auc, best_threshold_bound
    perfect = [(1, False), (1, False), (0, True), (0, True)]
    assert _auc(perfect) == 1.0
    assert _auc([(0.5, False), (0.5, True)]) == 0.5
    assert best_threshold_bound(perfect) == 1.0
    # Every threshold that rejects a wrong stop also rejects >5% of correct ones.
    assert best_threshold_bound([(1, False), (1, True), (0, True)]) == 0.0
