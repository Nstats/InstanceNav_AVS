#!/usr/bin/env python3
"""Manual, one-experiment launcher for the planned InstanceNav ablations.

--dry-run prints only; --check-only imports dependencies but never navigates or
calls a model. Normal execution archives the configuration and owned runtime
source, then delegates to run_instance100.sh. There is deliberately no 'all'.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
# The queue script only lists which launcher commands to run next. Editing it to queue
# more runs must not change the fingerprint of the code under test, so it is neither
# hashed nor archived. Nothing it contains reaches the navigation process.
FINGERPRINT_EXCLUDE = {'scripts/run_continuous_evaluation.sh'}
# Runs are compared only on the algorithm core: the navigation entry point, the packages it
# imports, its configuration, the task subset and the submodule versions. Launchers, tools,
# tests, docs and environment files are archived and hashed in every manifest, but editing
# them (or syncing them from another machine) never blocks a run or a resume.
CORE_TOP_LEVEL = {'PSG_Nav.py'}
CORE_PACKAGES = {'utils', 'configs', 'graph'}
DATA_FILES = ('data/datasets/instancenav/val10/val10.json.gz',
              'data/datasets/instancenav/val10/val10.manifest.json')
FULL_CONFIG = 'configs/runtime/instancenav_val_full_1000atomic_1m.yaml'
FULL_DATA_FILES = ('data/datasets/instancenav/val_full/val_full.json.gz',
                   'data/datasets/instancenav/val_full/val_full.manifest.json')
COMMON = {
    'LANDMARK_COMPARISON_REPEATS': '1', 'LANDMARK_COMPARISON_WORKERS': '10',
    'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'false',
    'ENABLE_OBJECT_RELATIONS': 'false', 'ENABLE_GROUP_LLM_FILTER': 'true',
    'ENABLE_ROOM_LLM_FILTER': 'true', 'ENABLE_INSTANCE_CONFIRMATION': 'true',
    'INSTANCE_CONFIRMATION_THRESHOLD': '1', 'VERIFY_BEFORE_CANDIDATE': 'false',
    'INSTANCE_CONFIRMATION_COUNT_UNIT': 'frame',
    'INSTANCE_CONFIRMATION_MODE': 'positive_match',
    'INSTANCE_VERIFICATION_PROMPT': 'match_strict_no',
    'INSTANCE_VERIFICATION_SCOPE': 'instance', 'INSTANCE_REJECTION_POLICY': 'none',
    'INSTANCE_REJECTION_SCOPE': 'episode', 'ENABLE_CANDIDATE_APPROACH': 'true',
    'INSTANCE_VERIFICATION_RECORD_INPUTS': 'true',
    'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
    'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false',
    'ENABLE_RELATIVE_EVIDENCE': 'false', 'RAG_MODE': '0', 'VLM': 'qwen3.5-flash',
}
VARIANTS = {
    'B': {},
    'E1': {'ENABLE_INSTANCE_CONFIRMATION': 'false'},
    'E2': {'INSTANCE_VERIFICATION_SCOPE': 'category'},
    'E3': {'INSTANCE_REJECTION_POLICY': 'no'},
    'E4': {'INSTANCE_REJECTION_POLICY': 'non_yes'},
    'E5': {'INSTANCE_CONFIRMATION_THRESHOLD': '2'},
    # M7 (paper Table A, how far one negative answer reaches): a no/unknown releases the
    # candidate for the current approach only; it is not excluded (B: this frame only,
    # E3/E4: the whole episode).
    'T0': {'INSTANCE_REJECTION_POLICY': 'non_yes', 'INSTANCE_REJECTION_SCOPE': 'approach'},
    'T0no': {'INSTANCE_REJECTION_POLICY': 'no', 'INSTANCE_REJECTION_SCOPE': 'approach'},
    # M7 (paper Table C, which candidates are approached): none (A0) or only those the
    # VLM admits from where they are first seen (RA); B approaches every detection.
    'A0': {'ENABLE_CANDIDATE_APPROACH': 'false'},
    'A0VisibleEvidence': {'ENABLE_CANDIDATE_APPROACH': 'false',
                          'INSTANCE_VERIFICATION_PROMPT': 'match_visible_evidence'},
    'RA': {'VERIFY_BEFORE_CANDIDATE': 'true'},
    # A0 is the paper's main configuration (2026-11-19). These add one E-variant rule back
    # on top of A0 instead of B, so Table 4 can be read against the main configuration.
    'A0E3': {'ENABLE_CANDIDATE_APPROACH': 'false', 'INSTANCE_REJECTION_POLICY': 'no'},
    'A0E4': {'ENABLE_CANDIDATE_APPROACH': 'false', 'INSTANCE_REJECTION_POLICY': 'non_yes'},
    'A0E5': {'ENABLE_CANDIDATE_APPROACH': 'false', 'INSTANCE_CONFIRMATION_THRESHOLD': '2'},
}
# Model behind every call (exploration and verification), as run_instance100.sh --vlm accepts it.
# The default keeps the existing presets unchanged; another VLM is a separate comparison batch.
VLMS = ('qwen3.5-flash', 'gemini-2.5-flash', 'qwen3-vl-8b-instruct')
# Preserve the historical settings identity (missing NAV_BASE = PSG-Nav) for resume.
NAV_BASES = ('psgnav', 'vlfm')
# Optional S6 flags are materialized when enabled, not added to historical A0 manifests.
RELATIVE_OPTIONS = {
    'RELATIVE_EVIDENCE_CALIBRATION': 'configs/relative_evidence_s6.json',
    'RELATIVE_EVIDENCE_USE_U': 'true', 'RELATIVE_EVIDENCE_EARLY_STOP': 'true',
    'RELATIVE_EVIDENCE_PAIRWISE': 'true', 'RELATIVE_EVIDENCE_DROP_REJECTED': 'true',
}


def flag(key):
    return '--' + key.lower().replace('_', '-')


def bool_value(value):
    for canonical, aliases in [('true', ('true', '1', 'yes')), ('false', ('false', '0', 'no'))]:
        if value.lower() in aliases:
            return canonical
    raise argparse.ArgumentTypeError('expected true/false/1/0/yes/no')


def common(vlm=COMMON['VLM']):
    if vlm not in VLMS:
        raise ValueError(f'VLM must be one of {", ".join(VLMS)}')
    return {**COMMON, 'VLM': vlm}


def settings(experiment, vlm=COMMON['VLM'], nav_base='psgnav', overrides=None):
    if nav_base not in NAV_BASES:
        raise ValueError(f'nav base must be one of {", ".join(NAV_BASES)}')
    overrides = overrides or {}
    resolved = {**common(vlm), **VARIANTS[experiment], **overrides}
    if 'INSTANCE_VERIFICATION_PROMPT' in overrides and (
            resolved['ENABLE_INSTANCE_CONFIRMATION'] != 'true'
            or resolved['INSTANCE_CONFIRMATION_MODE'] != 'positive_match'
            or resolved['INSTANCE_VERIFICATION_SCOPE'] != 'instance'):
        print('Note: --instance-verification-prompt is inactive without positive_match instance verification',
              file=sys.stderr)
    if nav_base == 'vlfm':
        # VLFM uses value-map exploration, not PSG-Nav's landmark/room filters.
        for key in ('LANDMARK_COMPARISON_REPEATS', 'LANDMARK_COMPARISON_WORKERS',
                    'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION', 'ENABLE_OBJECT_RELATIONS',
                    'ENABLE_GROUP_LLM_FILTER', 'ENABLE_ROOM_LLM_FILTER'):
            if key in overrides and resolved[key] != COMMON[key]:
                raise ValueError(f'{flag(key)} changes PSG-Nav exploration, not VLFM')
    if resolved['ENABLE_RELATIVE_EVIDENCE'] == 'true':
        resolved = {**RELATIVE_OPTIONS, **resolved}
    if nav_base != 'psgnav':
        resolved['NAV_BASE'] = nav_base
    return resolved


def launch_command(resolved, gpus, timestamp=None, resume=False, *, full=False):
    cmd = ['bash', str(ROOT / 'scripts/run_instance100.sh')]
    for key, value in resolved.items():
        cmd.extend([flag(key), value])
    if 'NAV_BASE' not in resolved:
        cmd.extend(['--nav-base', 'psgnav'])
    if full:
        cmd.append('--full')
    cmd.extend(['--gpus', gpus])
    if timestamp:
        cmd.extend(['--timestamp', timestamp])
    if resume:
        cmd.append('--resume')
    return cmd


def source_inventory(nav_base='psgnav', *, full=False):
    """Archive application code/config, not credentials, datasets or model weights."""
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    paths = set()
    for name in filter(None, names):
        p = Path(name)
        if name in FINGERPRINT_EXCLUDE or (name == FULL_CONFIG and not full):
            continue
        vlfm_source = (nav_base == 'vlfm' and
                       (p.parts[0] == 'nav_bases' or p.parts[:2] == ('third_party', 'vlfm')) and
                       (p.suffix in {'.py', '.md', '.txt', '.json', '.sha256'} or
                        p.name.startswith('LICENSE') or p.name == 'SOURCES'))
        if (vlfm_source or (len(p.parts) == 1 and p.suffix in {'.py', '.yml', '.yaml'}) or
                (p.parts[0] in {'utils', 'configs', 'graph', 'scripts', 'tools'}
                 and p.suffix in {'.py', '.sh', '.json', '.yaml', '.yml'})):
            if (ROOT / p).is_file():
                paths.add(name)
    # The subset itself is hashed, not archived (large data remains external).
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
              for n in sorted(paths | set(FULL_DATA_FILES if full else DATA_FILES))}
    submodules = subprocess.check_output(['git', 'submodule', 'status', '--recursive'], cwd=ROOT).decode()
    digest = hashlib.sha256(json.dumps({'files': hashes, 'submodules': submodules},
                                    sort_keys=True).encode()).hexdigest()
    return sorted(paths), hashes, submodules, digest


def is_core(name):
    parts = Path(name).parts
    return name in CORE_TOP_LEVEL or name in DATA_FILES or name in FULL_DATA_FILES or (len(parts) > 1 and parts[0] in CORE_PACKAGES)


def core_digest(hashes, submodules):
    """Digest of the algorithm core only, computed from per-file hashes (also old manifests)."""
    core = {name: value for name, value in hashes.items() if is_core(name)}
    return hashlib.sha256(json.dumps({'files': core, 'submodules': submodules},
                                     sort_keys=True).encode()).hexdigest()


def _manifest_core(manifest):
    if 'core_digest' in manifest:
        return manifest['core_digest']
    return core_digest(manifest.get('files_sha256', {}), manifest.get('submodule_versions', ''))


def _suite_core(suite_path, suite):
    """Suites written before core digests existed are identified by any of their manifests."""
    if 'core_digest' in suite:
        return suite['core_digest']
    for manifest in sorted(suite_path.parent.glob('*/manifest.json')):
        return _manifest_core(json.loads(manifest.read_text()))
    return None


def archive_run(args, resolved, cmd, directory, inventory):
    paths, hashes, submodules, digest = inventory
    core = core_digest(hashes, submodules)
    manifest_path = directory / 'manifest.json'
    python = str(Path(sys.executable).resolve())
    # Keep the originating preset for provenance, but compare the actual settings:
    # an alias with the same final values describes the same run.
    identity = {'experiment': args.experiment, 'settings': resolved, 'timestamp': args.timestamp,
                'dataset': 'instancenav', 'nav_base': resolved.get('NAV_BASE', 'psgnav')}
    if getattr(args, 'full', False):
        identity.update(task_set='full_val', dataset_config=FULL_CONFIG, total=1000)
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError('Resume requires this launcher\'s original manifest')
        prior = json.loads(manifest_path.read_text())
        # Older manifests encode the base only inside settings (absent = PSG-Nav).
        if (prior.get('settings') != resolved or prior.get('timestamp') != args.timestamp
                or prior.get('task_set', 'val10') != identity.get('task_set', 'val10')
                or prior.get('dataset', 'instancenav') != identity['dataset']
                or prior.get('nav_base', identity['nav_base']) != identity['nav_base']):
            raise ValueError('Resume settings changed; restore the original run settings')
        if _manifest_core(prior) != core:
            raise ValueError('Resume algorithm core changed (PSG_Nav.py, utils/, configs/, graph/, '
                             'task subset or submodules); restore it to resume this run')
        return manifest_path
    if directory.exists() or (ROOT / f'data/results_instancenav_{args.timestamp}').exists():
        raise ValueError('Timestamp already exists; use --resume or a new timestamp')
    suite_path = ROOT / 'outputs/instance_ablation_runs' / args.suite / 'suite.json'
    if suite_path.exists():
        suite = json.loads(suite_path.read_text())
        if suite.get('common') != common(args.vlm):
            raise ValueError('Suite common settings (including the VLM) differ; select a new --suite')
        expected = _suite_core(suite_path, suite)
        if expected is not None and expected != core:
            raise ValueError('Suite algorithm core changed (PSG_Nav.py, utils/, configs/, graph/, '
                             'task subset or submodules); restore it or select a new --suite')
    directory.mkdir(parents=True)
    if not suite_path.exists():
        suite_path.write_text(json.dumps({'core_digest': core, 'source_digest': digest, 'python': python,
                                          'common': common(args.vlm)}, indent=2) + '\n')
    identity.update({'source_digest': digest, 'core_digest': core, 'python': python, 'gpus': args.gpus})
    with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
        for name in paths:
            archive.add(ROOT / name, arcname=name, recursive=False)
    manifest = {**identity, 'command': cmd, 'files_sha256': hashes,
                'submodule_versions': submodules, 'python_version': sys.version,
                'archive_scope': 'Owned application source/config; dependency and dataset paths remain external.',
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('experiment', choices=VARIANTS, help='starting preset; explicit navigation flags override it')
    parser.add_argument('--full', action='store_true', help='run the full official val split instead of the frozen 100')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--timestamp')
    parser.add_argument('--suite', default='avs5_20261110')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--vlm', default=COMMON['VLM'], choices=VLMS,
                        help='model for all calls; part of the run settings and the suite identity')
    parser.add_argument('--nav-base', default='psgnav', choices=NAV_BASES,
                        help='navigation base; VLFM checks its capabilities against the final flags')
    switches = {**COMMON, **RELATIVE_OPTIONS}
    del switches['VLM']  # already handled above, including suite identity
    group = parser.add_argument_group('navigation overrides (applied after the preset)')
    for key, default in switches.items():
        value_type = bool_value if default in ('true', 'false') else (
            str if key == 'RELATIVE_EVIDENCE_CALIBRATION' else str.lower)
        group.add_argument(flag(key), dest=key, type=value_type, metavar='VALUE')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for value in (args.suite, args.timestamp):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('suite/timestamp must contain only letters, digits, underscore or hyphen')
    gpu_ids = args.gpus.split(',')
    if not re.fullmatch(r'\d+(,\d+)*', args.gpus) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('--gpus requires one or more distinct IDs, e.g. 0 or 0,1')
    if not args.timestamp and (args.resume or not (args.check_only or args.dry_run)):
        parser.error('--timestamp is required for evaluation/resume')
    overrides = {key: getattr(args, key) for key in switches if getattr(args, key) is not None}
    resolved = settings(args.experiment, args.vlm, args.nav_base, overrides)
    cmd = launch_command(resolved, args.gpus, args.timestamp, args.resume, full=args.full)
    env = {**os.environ, **resolved, 'PSGNAV_PYTHON': sys.executable,
           'NAV_BASE': args.nav_base, 'INSTANCE_ABLATION_ID': args.experiment}
    for legacy in ('ENABLE_CONTRADICTION_CHECK', 'ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL'):
        env.pop(legacy, None)
    if args.dry_run:
        result = subprocess.run([*cmd, '--validate-settings-only'], env=env, cwd=ROOT)
        if result.returncode:
            return result.returncode
        print(json.dumps({'experiment': args.experiment, 'dataset': 'instancenav', 'nav_base': args.nav_base,
                          'settings': resolved, 'command': cmd}, indent=2))
        return 0
    print('Experiment:', args.experiment, '\nCommand:', shlex.join(cmd), flush=True)
    result = subprocess.run([*cmd, '--check-only'], env=env, cwd=ROOT)
    if result.returncode or args.check_only:
        return result.returncode
    directory = ROOT / 'outputs/instance_ablation_runs' / args.suite / args.timestamp
    manifest = archive_run(args, resolved, cmd, directory, source_inventory(args.nav_base, full=args.full))
    if args.resume:
        env['INSTANCE_ABLATION_ID'] = json.loads(manifest.read_text())['experiment']
    env['INSTANCE_ABLATION_MANIFEST'] = str(manifest)
    print('Manifest:', manifest, flush=True)
    os.chdir(ROOT)
    os.execvpe('bash', cmd, env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc))
