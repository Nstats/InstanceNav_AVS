import random

import pytest

from tools.analyze_verification_set import (
    AREA_BINS, INST_BINS, bin_of, binom_tail, classify, cluster_bootstrap, first_yes_target_prob,
    kaplan_meier, km_survival_at, lag1_shuffle_test, lag1_stats, lock_identity, macro_rate, pooled_rate,
)


def test_bin_of_half_open_and_instance_bins():
    assert bin_of(0.0, AREA_BINS) == '<2%'
    assert bin_of(0.02, AREA_BINS) == '2-5%'
    assert bin_of(0.2, AREA_BINS) == '>=20%'
    assert bin_of(1.0, AREA_BINS) == '>=20%'
    assert bin_of(None, AREA_BINS) is None
    assert [bin_of(n, INST_BINS) for n in (1, 2, 3, 4, 7, 8, 40)] == ['1', '2-3', '2-3', '4-7', '4-7', '>=8', '>=8']


def test_macro_vs_pooled_rates():
    pairs = {'a': (1, 1), 'b': (0, 99)}          # one episode dominates the pooled rate
    assert macro_rate(pairs) == pytest.approx(0.5)
    assert pooled_rate(pairs) == pytest.approx(0.01)
    assert macro_rate({'a': (0, 0)}) is None


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = random.Random(1)
    pairs = {i: (rng.randint(0, 5), 5) for i in range(40)}
    lo, hi = cluster_bootstrap(pairs, macro_rate, n_boot=500, seed=3)
    assert (lo, hi) == cluster_bootstrap(pairs, macro_rate, n_boot=500, seed=3)
    assert lo <= macro_rate(pairs) <= hi


def test_kaplan_meier_handles_censoring():
    # Events at 1 and 3; one subject censored at 2 leaves 2 at risk at t=3.
    curve = kaplan_meier([(1, True), (2, False), (3, True), (4, False)])
    assert curve == [(1, pytest.approx(0.75)), (3, pytest.approx(0.375))]
    assert km_survival_at(curve, 0) == 1.0
    assert km_survival_at(curve, 2) == pytest.approx(0.75)
    assert km_survival_at(curve, 10) == pytest.approx(0.375)


def test_lag1_stats_and_shuffle_detects_bursts():
    # Pairs (1,1), (1,0), (0,0): P(y|y) = 1/2, P(y|n) = 0/1.
    assert lag1_stats([[1, 1, 0, 0]]) == (0.5, 0.0, 2, 1)
    rng = random.Random(0)
    bursty = [[1] * 4 + [0] * 8 + [1] * 4 + [0] * 8 for _ in range(30)]
    iid = [[int(rng.random() < 1 / 3) for _ in range(24)] for _ in range(30)]
    assert lag1_shuffle_test(bursty, n_perm=200)['p'] < 0.05
    assert lag1_shuffle_test(iid, n_perm=200)['p'] > 0.05


def test_first_yes_target_prob():
    assert first_yes_target_prob([(False, 0.5), (True, 1.0)]) == pytest.approx(0.5)
    assert first_yes_target_prob([(True, 0.2)]) == pytest.approx(1.0)
    assert first_yes_target_prob([(False, 0.0)]) is None


def test_binom_tail():
    assert binom_tail(3, 0.5, 0) == pytest.approx(1.0)
    assert binom_tail(3, 0.5, 3) == pytest.approx(0.125)
    assert binom_tail(2, 0.1, 1) == pytest.approx(0.19)


def _box(step, label, gps, conf='yes', inst=7):
    return {'step': step, 'label': label, 'est_gps': gps, 'confirmation': conf, 'admission': None,
            'gt_instance': inst, 'box_index': 0}


def _episode(success=0.0, termination='stop_called', locks=(), boxes=(), target_px=()):
    exposure = [{'step': s, 'target_px': px, 'same_category_px': {}} for s, px in target_px]
    ep = {'success': success, 'termination': termination, 'locks': [], 'boxes': list(boxes),
          'exposure': exposure, 'target_instance': 5, 'same_category_instances': {}}
    for L in locks:
        L = dict(L)
        L['identity'], L['identity_instance'], L['identity_source'] = lock_identity(ep, L)
        ep['locks'].append(L)
    return ep


def test_lock_identity_uses_the_box_at_the_lock():
    ep = _episode(boxes=[_box(10, 'same_category', [1.0, 2.0]), _box(10, 'target', [3.0, 3.0], inst=5)])
    assert lock_identity(ep, {'step': 10, 'goal_gps': [1.0, 2.0]})[0] == 'same_category'
    assert lock_identity(ep, {'step': 10, 'goal_gps': [3.0, 3.0]})[:2] == ('target', 5)
    # No matching box: fall back to the nearest same-category instance.
    fb = {'step': 11, 'goal_gps': [9.0, 9.0], 'nearest_instance': 4, 'nearest_instance_dist_m': 0.4, 'nearest_is_target': False}
    assert lock_identity(ep, fb)[:2] == ('same_category', 4)
    fb.update(nearest_instance_dist_m=2.0)
    assert lock_identity(ep, fb)[0] == 'other'


def test_classify_wrong_stops_by_visibility():
    lock = {'step': 50, 'goal_gps': [1.0, 2.0]}
    box = _box(50, 'same_category', [1.0, 2.0])
    before = _episode(locks=[lock], boxes=[box], target_px=[(10, 0), (60, 5000)])
    after = _episode(locks=[lock], boxes=[box], target_px=[(10, 5000)])
    assert classify(before, 2000) == 'ws_samecat_before'
    assert classify(after, 2000) == 'ws_samecat_after'
    other = _episode(locks=[lock], boxes=[_box(50, 'other', [1.0, 2.0])])
    assert classify(other, 2000) == 'ws_other'
    assert classify(_episode(success=1.0), 2000) == 'success'


def test_classify_step_limit_chain():
    lim = dict(termination='step_limit')
    assert classify(_episode(**lim, target_px=[(1, 100)]), 2000) == 'lim_never_visible'
    assert classify(_episode(**lim, target_px=[(1, 3000)]), 2000) == 'lim_not_boxed'
    rejected = _box(1, 'target', [0.0, 0.0], conf='no', inst=5)
    assert classify(_episode(**lim, target_px=[(1, 3000)], boxes=[rejected]), 2000) == 'lim_not_accepted'
    accepted = _box(1, 'target', [0.0, 0.0], conf='yes', inst=5)
    assert classify(_episode(**lim, target_px=[(1, 3000)], boxes=[accepted]), 2000) == 'lim_accepted_no_lock'
    tl = _episode(**lim, locks=[{'step': 1, 'goal_gps': [0.0, 0.0]}], boxes=[accepted], target_px=[(1, 3000)])
    assert classify(tl, 2000) == 'lim_target_locked'
