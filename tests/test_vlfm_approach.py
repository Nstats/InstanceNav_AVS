"""Mocked candidate decisions; no navigator, simulator, model server or oracle IDs."""
from unittest.mock import Mock

import numpy as np
import pytest

from nav_bases.avs_decision import Proposal
from nav_bases.vlfm_approach import ApproachDecision


def proposal(x, distance, pixel=0):
    return Proposal([pixel, 0, pixel + 10, 10], np.array([[x, 1., .2, 1.]]), distance)


def test_ra_batches_admission_then_only_admitted_near_confirmation():
    verify = Mock(side_effect=[['yes', 'no', 'yes', 'yes', 'unknown', 'yes'],
                               ['yes', 'no', 'yes']])
    log = Mock()
    decision = ApproachDecision(verify, admission=True, log=log)
    frame = [Proposal([0] * 4, None, None), proposal(6, 6, 10), proposal(1, 1.3, 20),
             proposal(2, 2.2, 30), proposal(3, 2.5, 40), proposal(7, 7, 50), proposal(4, 3.8, 60)]
    rgb, context = object(), {'step': 11, 'episode': 2}
    assert decision.select(frame, rgb, context) == [3, 6]
    for call, stage, indices in zip(verify.call_args_list, ['candidate_admission', 'goal_confirmation'],
                                    [[1, 2, 3, 4, 5, 6], [3, 4, 6]]):
        assert call.args == (rgb, [frame[i].box for i in indices],
                             {**context, 'stage': stage, 'box_indices': indices})
    assert decision.verification_calls == verify.call_count == 2
    assert decision.commit_count == 1 and decision.approach_goal is None
    assert [id(c) for c in decision.committed] == [id(frame[i].cloud) for i in (3, 6)]
    assert [c.args[0] for c in log.call_args_list] == [
        'vlfm_candidate_admission', 'vlfm_goal_confirmation',
        'vlfm_candidate_vote', 'vlfm_candidate_vote', 'vlfm_commit']


@pytest.mark.parametrize('answer', ['no', 'unknown'])
def test_ra_admission_neither_votes_nor_overrides_failed_confirmation(answer):
    verify = Mock(side_effect=[['yes', answer], [answer], ['yes'], ['yes'], ['yes'], ['yes']])
    decision = ApproachDecision(verify, admission=True, threshold=2)
    near = proposal(1, 2)
    assert decision.select([near, proposal(2, 7)], None, {'step': 0}) == []
    assert decision.approach_goal is None
    assert decision.select([near], None, {'step': 1}) == []
    np.testing.assert_array_equal(decision.approach_goal, [1, 1])
    assert decision.select([near], None, {'step': 2}) == [0]


@pytest.mark.parametrize('threshold', [1, 2])
def test_nearest_current_kept_and_no_new_goal_on_failed_or_missing_frame(threshold):
    verify = Mock(side_effect=[['no', 'unknown'], ['unknown'], ['no']])
    log = Mock()
    decision = ApproachDecision(verify, threshold=threshold, log=log)
    frame = [proposal(9, 7), proposal(1, 1.3), proposal(5, 4.2), proposal(2, 2)]
    assert decision.select(frame, None, {'step': 0}) == []
    np.testing.assert_array_equal(decision.approach_goal, [5, 1])
    assert verify.call_args.args[2]['box_indices'] == [1, 3]
    for step, missing in enumerate(([], [Proposal([0] * 4, np.empty((0, 4)), None)], [proposal(3, 1.5)]), 1):
        assert decision.select(missing, None, {'step': step}) == []
        np.testing.assert_array_equal(decision.approach_goal, [5, 1])
    # Select from the current kept candidates, not the possibly closer old goal.
    assert decision.select([proposal(9, 7), proposal(3, 2)], None, {'step': 4}) == []
    np.testing.assert_array_equal(decision.approach_goal, [9, 1])
    assert verify.call_count == 3 and not decision.is_committed
    assert 'vlfm_approach_switch' in [c.args[0] for c in log.call_args_list]


@pytest.mark.parametrize('admission', [False, True])
def test_exact_four_metres_is_far_and_only_near_confirmation_votes(admission):
    verify = Mock(return_value=['yes'])
    decision = ApproachDecision(verify, admission=admission, threshold=2)
    for step in (0, 1):
        assert decision.select([proposal(1, 4.)], None, {'step': step}) == []
        np.testing.assert_array_equal(decision.approach_goal, [1, 1])
    assert [c.args[2]['stage'] for c in verify.call_args_list] == ['candidate_admission'] * (2 * admission)
    assert decision.select([proposal(1, 3.999)], None, {'step': 2}) == []
    assert decision.select([proposal(1, 3.8)], None, {'step': 3}) == [0]


def test_reprocessing_or_duplicate_boxes_cannot_supply_an_extra_frame():
    verify = Mock(return_value=['yes', 'yes'])
    log = Mock()
    decision = ApproachDecision(verify, threshold=2, log=log)
    for step, expected in [(11, []), (11, []), (17, [0, 1])]:
        assert decision.select([proposal(2, 3), proposal(2, 3)], None, {'step': step}) == expected
    votes = [c.kwargs for c in log.call_args_list if c.args == ('vlfm_candidate_vote',)]
    assert [(v['context']['step'], v['count']) for v in votes] == [(11, 1), (17, 2)]


def test_close_independent_candidates_cannot_pool_votes_or_enter_unqualified():
    verify = Mock(side_effect=[['yes', 'no'], ['yes', 'no'], ['yes', 'yes', 'yes']])
    decision = ApproachDecision(verify, threshold=2)
    left, right, new = proposal(2, 2.1, 0), proposal(2.3, 2.3, 20), proposal(7, 3.8, 50)
    assert decision.select([left, right], None, {'step': 1}) == []
    assert decision.select([right, left], None, {'step': 2}) == []
    np.testing.assert_array_equal(decision.approach_goal, [2.3, 1])
    assert decision.select([right, new, left], None, {'step': 3}) == [0, 2]
    assert decision.approach_goal is None


def test_new_neighbour_cannot_take_a_still_visible_candidates_vote():
    verify = Mock(side_effect=[['yes'], ['yes', 'no'], ['yes']])
    decision = ApproachDecision(verify, threshold=2)
    old, new = proposal(2, 2, 0), proposal(2.3, 2.3, 20)
    assert decision.select([old], None, {'step': 1}) == []
    assert decision.select([new, old], None, {'step': 2}) == []
    assert decision.select([new], None, {'step': 3}) == [0]


@pytest.mark.parametrize('answer', ['no', 'unknown'])
def test_failed_neighbour_never_moves_an_old_vote_towards_a_new_box(answer):
    verify = Mock(side_effect=[['yes'], [answer, 'yes'], ['yes'], ['yes']])
    decision = ApproachDecision(verify, threshold=2)
    old = proposal(0, 2)
    assert decision.select([old], None, {'step': 1}) == []
    assert decision.select([proposal(.6, 2, 20), proposal(6, 2, 40)], None, {'step': 2}) == []
    # This is within .8 m of the failed box, but not the original vote.
    assert decision.select([proposal(1.25, 2, 20)], None, {'step': 3}) == []
    assert decision.select([old], None, {'step': 4}) == [0]


def test_a_near_yes_keeps_its_track_ahead_of_a_closer_far_box():
    verify = Mock(return_value=['yes'])
    decision = ApproachDecision(verify, threshold=2)
    assert decision.select([proposal(2, 3)], None, {'step': 1}) == []
    frame = [proposal(2, 4.1, 20), proposal(2.1, 3.7, 0)]
    assert decision.select(frame, None, {'step': 2}) == [1]


@pytest.mark.parametrize('reason, expected', [('reached', [0]), ('stuck', [])])
def test_only_stuck_release_erases_votes_including_for_other_tracks(reason, expected):
    verify = Mock(side_effect=[['yes', 'yes'], ['no'], ['yes']])
    decision = ApproachDecision(verify, threshold=2)
    left, right = proposal(1, 1.8), proposal(5, 3.8, 20)
    assert decision.select([left, right], None, {'step': 1}) == []
    np.testing.assert_array_equal(decision.approach_goal, [1, 1])
    assert decision.select([], None, {'step': 2}) == []
    assert decision.select([right], None, {'step': 3}) == []
    decision.release_approach(reason, {'step': 4})
    assert decision.approach_goal is None
    assert decision.select([right], None, {'step': 5}) == expected


def test_committed_updates_use_a0_and_map_withdrawal_requires_fresh_votes():
    verify = Mock(return_value=['yes'])
    decision = ApproachDecision(verify, threshold=2)
    near = proposal(2, 3.5)
    assert decision.select([near], None, {'step': 1}) == []
    assert decision.select([near], None, {'step': 2}) == [0]
    assert decision.select([proposal(7, 1.3), proposal(2.2, 5)], None, {'step': 3}) == [1]
    assert verify.call_count == 2 and decision.approach_goal is None
    decision.withdraw('target_map_cleared', {'step': 4})
    assert not decision.is_committed
    assert decision.select([near], None, {'step': 5}) == []
    assert decision.select([near], None, {'step': 6}) == [0]
    assert decision.commit_count == 2


def test_reset_discards_episode_track_and_approach_state():
    decision = ApproachDecision(Mock(return_value=['yes']), threshold=2)
    near = proposal(2, 2)
    assert decision.select([near], None, {'step': 8}) == []
    decision.reset()
    assert decision.approach_goal is None and not decision.is_committed
    assert decision.verification_calls == decision.commit_count == 0
    assert decision.select([near], None, {'step': 0}) == []
    assert decision.select([near], None, {'step': 1}) == [0]
    decision.reset()
    assert decision.commit_count == 0 and not decision.is_committed
