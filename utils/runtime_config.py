"""Per-process Habitat runtime config: private copy, split injection, load check.

Two workers of one run used to write the same /tmp/psgnav_runtime_configs/<stem>.<split>.yaml.
``Path.write_text`` truncates before writing, so a worker loading while the other was
writing read an empty file; ``habitat.get_config`` silently fell back to Habitat defaults
(no atomic actions, 90 deg depth at 1.25 m, normalized depth). Each process now writes
its own file atomically, and the loaded config is compared with the base yaml before use.
"""
import atexit
import hashlib
import os
import tempfile
from pathlib import Path

import yaml

RUNTIME_DIR = Path('/tmp/psgnav_runtime_configs')

# Values recorded in the run header so a log shows the geometry the policy really used.
SUMMARY_KEYS = (
    'ENVIRONMENT.MAX_EPISODE_STEPS',
    'ENVIRONMENT.ATOMIC_ACTIONS',
    'SIMULATOR.TURN_ANGLE',
    'SIMULATOR.AGENT_0.HEIGHT',
    'SIMULATOR.AGENT_0.RADIUS',
    'SIMULATOR.DEPTH_SENSOR.HFOV',
    'SIMULATOR.DEPTH_SENSOR.MIN_DEPTH',
    'SIMULATOR.DEPTH_SENSOR.MAX_DEPTH',
    'SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH',
    'SIMULATOR.DEPTH_SENSOR.POSITION',
    'TASK.SUCCESS_DISTANCE',
    'DATASET.SPLIT',
    'DATASET.DATA_PATH',
)


def inject_split(text, dataset_split, source='config'):
    """Return ``text`` with DATASET.SPLIT replaced; raise if the block has no SPLIT."""
    in_dataset_block = False
    replaced = False
    new_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == 'DATASET:':
            in_dataset_block = True
            new_lines.append(line)
            continue
        if in_dataset_block and stripped and not line.startswith('  '):
            in_dataset_block = False
        if in_dataset_block and stripped.startswith('SPLIT:'):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}SPLIT: {dataset_split}')
            replaced = True
            continue
        new_lines.append(line)
    if not replaced:
        raise ValueError(f'Failed to inject DATASET.SPLIT into config: {source}')
    return '\n'.join(new_lines) + '\n'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_runtime_config_file(base_config_file, dataset_split, runtime_dir=RUNTIME_DIR, remove_at_exit=True):
    """Write this process's own copy of the base config with DATASET.SPLIT set.

    The file name carries the pid, and the content is written to a temporary file in the
    same directory and renamed into place, so no reader can see a partial file.
    Returns (path, sha256).
    """
    base_path = Path(base_config_file)
    content = inject_split(base_path.read_text(), dataset_split, source=base_config_file)
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    final_path = runtime_dir / f'{base_path.stem}.{dataset_split}.{os.getpid()}{base_path.suffix}'
    fd, tmp_name = tempfile.mkstemp(dir=runtime_dir, prefix=f'.{final_path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, final_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    if remove_at_exit:
        atexit.register(_remove_quietly, final_path)
    return str(final_path), hashlib.sha256(content.encode()).hexdigest()


def _remove_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _flatten(node, prefix=''):
    for key, value in node.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from _flatten(value, name + '.')
        else:
            yield name, value


def _lookup(config, dotted):
    node = config
    for part in dotted.split('.'):
        if isinstance(node, dict):
            if part not in node:
                raise KeyError(dotted)
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            raise KeyError(dotted)
    return node


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def config_mismatches(config, base_config_file, dataset_split):
    """List (key, expected, loaded) for every base-yaml leaf the loaded config does not match.

    DATASET.SPLIT is expected to equal ``dataset_split``. An empty base file is itself an
    error: it would make every comparison vacuous.
    """
    expected = yaml.safe_load(Path(base_config_file).read_text())
    if not isinstance(expected, dict) or not expected:
        return [('<base config>', 'non-empty mapping', 'empty or invalid')]
    mismatches = []
    for key, want in _flatten(expected):
        if key == 'DATASET.SPLIT':
            want = dataset_split
        try:
            got = _lookup(config, key)
        except KeyError:
            mismatches.append((key, want, '<missing>'))
            continue
        if _normalize(got) != _normalize(want):
            mismatches.append((key, want, got))
    return mismatches


def verify_loaded_config(config, base_config_file, dataset_split):
    """Raise RuntimeError if the loaded Habitat config differs from the base yaml."""
    mismatches = config_mismatches(config, base_config_file, dataset_split)
    if mismatches:
        details = '; '.join(f'{key}: expected {want!r}, loaded {got!r}' for key, want, got in mismatches[:12])
        more = f' (+{len(mismatches) - 12} more)' if len(mismatches) > 12 else ''
        raise RuntimeError(
            f'Loaded Habitat config does not match {base_config_file}: {details}{more}. '
            'The runtime config file was probably empty or partially written; refusing to run.')


def assert_unchanged(path, digest):
    """Raise if the runtime file no longer has the content this process wrote."""
    current = file_sha256(path)
    if current != digest:
        raise RuntimeError(f'Runtime config {path} changed after it was written '
                           f'(sha256 {digest[:12]} -> {current[:12]}); refusing to run.')


def config_summary(config):
    """Small dict of key geometry/protocol values; missing keys are skipped."""
    summary = {}
    for key in SUMMARY_KEYS:
        try:
            value = _lookup(config, key)
        except KeyError:
            continue
        summary[key] = list(value) if isinstance(value, (list, tuple)) else value
    return summary
