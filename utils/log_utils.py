"""Centralized logging gates for PSG-Nav diagnostics.

Two kinds of output are distinguished:

1. Structured diagnostic records (always on): single-line JSON events with a
   stable ``[Tag]`` prefix, emitted via :func:`log_event`. These carry the
   per-step / per-candidate / per-episode evidence used for offline failure
   analysis and must not be gated.

2. Verbose chatter (off by default): legacy human-readable prints retained
   for interactive debugging. They are gated behind the ``PSGNAV_VERBOSE``
   environment variable, a comma-separated category whitelist, e.g.
   ``PSGNAV_VERBOSE=frontier,stuck,glip,landmark,detection``. Use ``all`` to
   re-enable everything.
"""

import json
import os
import subprocess

_CATEGORIES = {
    'frontier',   # per-cluster frontier rejection / connectivity chatter
    'stuck',      # per-step stuck recovery / stuck detection chatter
    'glip',       # GLIP tokens_positive / per-image inference time
    'landmark',   # per-landmark distance-to-frontier lines
    'detection',  # legacy [Goal Match] / [距离检查] / [GPS记录] lines
    'rag',        # RAG sliding-window chatter
    'planner',    # goal snap / goal priority / misc planner chatter
}

_raw = os.environ.get('PSGNAV_VERBOSE', '').strip().lower()
if _raw in ('all', '1', 'true', 'yes'):
    _enabled = set(_CATEGORIES)
else:
    _enabled = {c for c in (x.strip() for x in _raw.split(',')) if c in _CATEGORIES}


def verbose_enabled(category):
    """Return True when verbose prints for ``category`` are enabled."""
    return category in _enabled


def vprint(category, *args, **kwargs):
    """Print only when ``category`` is whitelisted in PSGNAV_VERBOSE."""
    if category in _enabled:
        print(*args, **kwargs)


def log_event(tag, **fields):
    """Emit one structured diagnostic record as a single stdout line.

    The line format is ``[Tag] {json}`` so log parsers can split on the
    first space. All values must be JSON-serializable; tuples become lists.
    """
    print(f'[{tag}] ' + json.dumps(fields, ensure_ascii=True, default=_json_default),
          flush=True)


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def code_version():
    """Best-effort code version string for the run header."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root,
            capture_output=True, text=True, timeout=5,
        )
        if head.returncode == 0:
            dirty = subprocess.run(
                ['git', 'status', '--porcelain'], cwd=root,
                capture_output=True, text=True, timeout=5,
            )
            suffix = '+dirty' if dirty.returncode == 0 and dirty.stdout.strip() else ''
            return f'git:{head.stdout.strip()}{suffix}'
    except (OSError, subprocess.SubprocessError):
        pass
    return 'git:unavailable'


def install_warning_filters():
    """Silence per-frame torch deprecation warnings from the detector stack.

    GLIP calls torch.utils.checkpoint and torch.cuda.amp.autocast on every
    inference frame, each emitting a UserWarning/FutureWarning. They carry no
    diagnostic value and accounted for ~60k lines per GPU log.
    """
    import warnings
    warnings.filterwarnings(
        'ignore', message=r'.*torch\.utils\.checkpoint.*use_reentrant.*',
        category=UserWarning)
    warnings.filterwarnings(
        'ignore', message=r'.*torch\.cuda\.amp\.autocast.*deprecated.*',
        category=FutureWarning)
