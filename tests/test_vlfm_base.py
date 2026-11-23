"""The VLFM base switch: AVS decision rules, and the PSG-Nav default path untouched."""
import ast
import os
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from nav_bases.avs_decision import AVSDecision, Proposal  # noqa: E402


def cloud(x, y, n=50):
    pts = np.zeros((n, 4))
    pts[:, 0], pts[:, 1], pts[:, 3] = x, y, 1
    return pts


def test_far_boxes_are_never_verified():
    verify = mock.Mock()
    avs = AVSDecision(verify)
    assert avs.select([Proposal([0, 0, 1, 1], cloud(5, 0), 4.5)], None) == []
    verify.assert_not_called()


def test_one_yes_commits_and_no_unknown_leave_no_trace():
    answers = iter([['no', 'unknown'], ['unknown', 'yes']])
    verify = mock.Mock(side_effect=lambda *a: next(answers))
    avs = AVSDecision(verify)
    frame = [Proposal([0, 0, 1, 1], cloud(1, 0), 1.5), Proposal([2, 2, 3, 3], cloud(3, 0), 3.0)]
    assert avs.select(frame, None) == []
    assert not avs.is_committed
    # The same boxes are verified again: a no/unknown is not remembered.
    assert avs.select(frame, None) == [1]
    assert avs.is_committed and verify.call_count == 2
    boxes = verify.call_args[0][1]
    assert len(boxes) == 2  # all eligible boxes in one request


def test_after_commit_only_the_same_instance_refines_without_requests():
    verify = mock.Mock(return_value=['yes'])
    avs = AVSDecision(verify, same_instance_m=0.8)
    avs.select([Proposal([0, 0, 1, 1], cloud(2, 0), 2.0)], None)
    later = [Proposal([0, 0, 1, 1], cloud(2.5, 0), 2.0), Proposal([0, 0, 1, 1], cloud(6, 0), 3.0)]
    assert avs.select(later, None) == [0]
    assert verify.call_count == 1


def test_withdraw_resumes_verification():
    verify = mock.Mock(return_value=['yes'])
    avs = AVSDecision(verify)
    avs.select([Proposal([0, 0, 1, 1], cloud(2, 0), 2.0)], None)
    avs.withdraw('target_map_cleared')
    assert not avs.is_committed
    avs.select([Proposal([0, 0, 1, 1], cloud(9, 0), 2.0)], None)
    assert verify.call_count == 2


def test_unusable_proposals_are_skipped():
    verify = mock.Mock()
    avs = AVSDecision(verify)
    assert avs.select([Proposal([0, 0, 1, 1], None, 0.5)], None) == []
    verify.assert_not_called()


def test_depth_normalisation_matches_vlfm_sensor():
    from nav_bases.vlfm_agent import normalise_depth
    depth = np.array([[0.2, 0.5], [2.75, 9.0]], dtype=np.float32)[..., None]
    np.testing.assert_allclose(normalise_depth(depth), [[0, 0], [0.5, 1]], atol=1e-6)


def test_instancenav_targets_map_to_vlfm_classes():
    from nav_bases.vlfm_agent import vlfm_target_name
    assert vlfm_target_name('instancenav', 'plant') == 'potted plant'
    assert vlfm_target_name('instancenav', 'tv_monitor') == 'tv'
    assert vlfm_target_name('coin', 'office_chair') == 'office chair'
    with pytest.raises(ValueError):
        vlfm_target_name('instancenav', 'lamp')


@pytest.mark.parametrize('env,ok', [
    ({'ENABLE_INSTANCE_CONFIRMATION': 'false'}, True),
    ({'ENABLE_CANDIDATE_APPROACH': 'false'}, True),
    ({'ENABLE_CANDIDATE_APPROACH': 'true'}, False),
    ({'ENABLE_CANDIDATE_APPROACH': 'false', 'INSTANCE_CONFIRMATION_THRESHOLD': '2'}, False),
    ({'ENABLE_CANDIDATE_APPROACH': 'false', 'INSTANCE_REJECTION_POLICY': 'no'}, False),
])
def test_vlfm_avs_settings_are_a0_only(env, ok):
    from nav_bases.vlfm_agent import validate_avs_settings
    keys = ['ENABLE_INSTANCE_CONFIRMATION', 'ENABLE_CANDIDATE_APPROACH', 'INSTANCE_CONFIRMATION_THRESHOLD',
            'INSTANCE_REJECTION_POLICY', 'VERIFY_BEFORE_CANDIDATE']
    clean = {k: v for k, v in os.environ.items() if k not in keys}
    with mock.patch.dict(os.environ, {**clean, 'VERIFY_BEFORE_CANDIDATE': 'false', **env}, clear=True):
        if ok:
            validate_avs_settings()
        else:
            with pytest.raises(ValueError):
                validate_avs_settings()


def test_psgnav_entry_point_does_not_reference_the_vlfm_base():
    source = (ROOT / 'PSG_Nav.py').read_text()
    assert 'nav_bases' not in source and 'vlfm' not in source.lower()
    for package in ('utils', 'configs', 'graph'):
        for path in (ROOT / package).rglob('*.py'):
            assert 'nav_bases' not in path.read_text(), path


def test_vlfm_entry_point_only_swaps_the_agent_class():
    tree = ast.parse((ROOT / 'VLFM_Nav.py').read_text())
    assigns = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    targets = {ast.unparse(t) for node in assigns for t in node.targets}
    assert targets == {"os.environ['NAV_BASE']", 'PSG_Nav.PSG_Nav_Agent'}


def _launcher_commands(*extra):
    env = {**os.environ, 'PSGNAV_PYTHON': '/bin/true'}
    script = ROOT / 'scripts/run_instance100.sh'
    text = script.read_text()
    return text, subprocess.run(['bash', '-n', str(script)], capture_output=True, env=env)


def test_launcher_defaults_to_psgnav():
    text, result = _launcher_commands()
    assert result.returncode == 0, result.stderr
    assert ': "${NAV_BASE:=psgnav}"' in text
    assert '  psgnav) nav_entry=PSG_Nav.py ;;' in text
    # Both evaluation commands go through the switch; no command hard-codes an entry point.
    commands = [line for line in text.splitlines()
                if '"$psgnav_python" -u' in line and 'tools/' not in line]
    assert len(commands) == 2 and all('"${nav_entry}"' in line for line in commands)


def test_launcher_rejects_unknown_base():
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_instance100.sh'), '--nav-base', 'sgnav',
                             '--check-only'], capture_output=True, text=True, cwd=ROOT,
                            env={**os.environ, 'PSGNAV_PYTHON': '/bin/true'})
    assert result.returncode == 2 and 'NAV_BASE must be psgnav or vlfm' in result.stderr


def _coin_launcher(tmp_path, script, *options):
    """Run a CoIN launcher with --check-only --full; a fake python records NAV_BASE."""
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text('#!/bin/bash\necho "$NAV_BASE" >> "$NAV_BASE_CAPTURE"\n'
                           'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('NAV_BASE', 'ENABLE_CANDIDATE_APPROACH',
                                                            'ENABLE_INSTANCE_CONFIRMATION')}
    env.update(PSGNAV_PYTHON=str(fake_python), NAV_BASE_CAPTURE=str(capture))
    result = subprocess.run(['bash', script, '--check-only', '--full', *options], env=env,
                            capture_output=True, text=True, cwd=ROOT)
    lines = capture.read_text().splitlines() if capture.exists() else []
    return result, lines[-1] if lines else None


def test_coin_launcher_defaults_to_psgnav(tmp_path):
    text = (ROOT / 'scripts/run_coin.sh').read_text()
    assert ': "${NAV_BASE:=psgnav}"' in text and '  psgnav) nav_entry=PSG_Nav.py ;;' in text
    commands = [line for line in text.splitlines() if '"$psgnav_python" -u' in line and 'tools/' not in line]
    assert commands == ['nav_cmd=("$psgnav_python" -u "$nav_entry" --dataset coin --dataset_split "$split" '
                        '--config "$config"']
    result, base = _coin_launcher(tmp_path, 'scripts/run_coin.sh')
    assert result.returncode == 0, result.stderr
    assert base == 'psgnav' and 'NAV_BASE: psgnav' in result.stdout


def test_coin_launcher_runs_vlfm_variants(tmp_path):
    result, base = _coin_launcher(tmp_path, 'scripts/run_coin.sh', '--nav-base', 'vlfm',
                                  '--enable-instance-confirmation', 'false')
    assert result.returncode == 0, result.stderr
    assert base == 'vlfm'
    result, base = _coin_launcher(tmp_path, 'scripts/run_coin_e0.sh', '--nav-base', 'vlfm')
    assert result.returncode == 0, result.stderr
    assert base == 'vlfm' and 'NAV_BASE: vlfm' in result.stdout


@pytest.mark.parametrize('options,message', [
    (['--nav-base', 'sgnav'], 'NAV_BASE must be psgnav or vlfm'),
    (['--nav-base', 'vlfm'], 'AVS A0 only'),  # CoIN defaults keep candidate approach on
])
def test_coin_launcher_rejects_bad_vlfm_settings(tmp_path, options, message):
    result, base = _coin_launcher(tmp_path, 'scripts/run_coin.sh', *options)
    assert result.returncode == 2 and message in result.stderr and base is None


def test_vlfm_policy_uses_official_settings_on_every_benchmark():
    source = (ROOT / 'nav_bases/vlfm_agent.py').read_text()
    assert 'settings = dict(VLFM_POLICY_DEFAULTS)\n' in source
    from nav_bases.vlfm_agent import VLFM_POLICY_DEFAULTS
    assert (VLFM_POLICY_DEFAULTS['pointnav_stop_radius'], VLFM_POLICY_DEFAULTS['non_coco_threshold'],
            VLFM_POLICY_DEFAULTS['max_obstacle_height'], VLFM_POLICY_DEFAULTS['agent_radius']) == (0.9, 0.4, 0.88, 0.18)


def test_detections_vlfm_discards_are_never_verified():
    from nav_bases.vlfm_agent import vlfm_range_ids
    # Mask at the image edge (too_offset): VLFM deletes the whole cloud, so no verification.
    assert vlfm_range_ids(True, [1.2, 1.3], 5.0) is None
    # No point within 95% of max depth: no lasting target either.
    assert vlfm_range_ids(False, [4.9, 5.0], 5.0) is None
    ids = vlfm_range_ids(False, [1.2, 4.9], 5.0)
    assert ids[0] == 1 and 0 < ids[1] < 1


def test_proposal_skips_what_vlfm_discards():
    source = (ROOT / 'nav_bases/vlfm_agent.py').read_text()
    body = source.split('def _proposal', 1)[1].split('return VLFMPolicy', 1)[0]
    assert 'vlfm_range_ids(too_offset(mask), local_cloud[:, 0], max_depth)' in body
    assert 'if within_range is None:\n                # VLFM keeps no point' in body


def test_ablation_launcher_nav_base():
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    from run_instance_ablation import launch_command, settings
    # PSG-Nav presets are unchanged: no NAV_BASE key, no --nav-base flag.
    assert 'NAV_BASE' not in settings('A0') and '--nav-base' not in launch_command('A0', '0,1', 't')
    assert settings('A0', nav_base='vlfm') == {**settings('A0'), 'NAV_BASE': 'vlfm'}
    cmd = launch_command('E1', '0', 't', nav_base='vlfm')
    assert cmd[cmd.index('--nav-base') + 1] == 'vlfm'
    with pytest.raises(ValueError):
        settings('B', nav_base='vlfm')


def test_queue_scripts_parse_and_name_vlfm_runs():
    # Queue files are named by benchmark; _h20 runs on the H20 with GPU 0 only.
    queues = {'run_continuous_evaluation_instancenav.sh': ('vlfm_A0_r1', 'vlfm_E1_r1'),
              'run_continuous_evaluation_coin.sh': (),
              'run_continuous_evaluation_coin_h20.sh': (),
              'run_continuous_evaluation_instancenav_h20.sh': ()}
    for name, runs in queues.items():
        path = ROOT / 'scripts' / name
        assert subprocess.run(['bash', '-n', str(path)]).returncode == 0
        commands = path.read_text().replace('\\\n', ' ').splitlines()
        active = [line.split() for line in commands if line.startswith('bash ') and '--timestamp' in line]
        vlfm_ids = [args[args.index('--timestamp') + 1] for args in active if 'vlfm' in args]
        assert sorted(vlfm_ids) == sorted(runs)
        if name.endswith('_h20.sh'):
            for args in active:
                flag = '--gpu' if '--gpu' in args else '--gpus'
                assert args[args.index(flag) + 1] == '0'


def test_local_vlfm_servers_bypass_the_proxy():
    from nav_bases.vlfm_agent import bypass_proxy_for_local_servers
    env = {'HTTP_PROXY': 'http://127.0.0.1:9981', 'no_proxy': 'example.org'}
    bypass_proxy_for_local_servers(env)
    assert env['NO_PROXY'] == 'localhost,127.0.0.1,::1'
    assert env['no_proxy'] == 'example.org,localhost,127.0.0.1,::1'
    assert env['HTTP_PROXY'] == 'http://127.0.0.1:9981'  # remote traffic keeps its proxy
    bypass_proxy_for_local_servers(env)
    assert env['no_proxy'] == 'example.org,localhost,127.0.0.1,::1'  # idempotent
