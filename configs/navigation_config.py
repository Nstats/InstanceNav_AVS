"""Navigation and instance-verification settings with explicit defaults.

The 100-episode launcher exports these switches for the navigator.
Instance confirmation selects positive matching or contradiction filtering.
Positive matching can optionally require a final multi-image contradiction check.
"""

import os
import warnings
from pathlib import Path


def _raw(name):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def env_int(name, default):
    """Positive integer from the environment, or the supplied default."""
    raw = _raw(name)
    if raw is None:
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def env_optional_int(name):
    """Positive integer, or None when the variable is unset."""
    if _raw(name) is None:
        return None
    return env_int(name, 1)


def env_bool(name, default):
    """Boolean from the environment. bool('False') is not used."""
    raw = _raw(name)
    if raw is None:
        return default
    token = raw.lower()
    if token in {"1", "true", "yes"}:
        return True
    if token in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true/false/1/0/yes/no, got {raw!r}")


def landmark_comparison_repeats():
    # The original pairwise loop compares each pair three times.
    return env_int("LANDMARK_COMPARISON_REPEATS", 3)


def landmark_comparison_workers():
    # None keeps the shared API concurrency already used by get_llm_responses.
    return env_optional_int("LANDMARK_COMPARISON_WORKERS")


def instance_description_for_exploration_enabled():
    """Use full instance descriptions in landmark selection; verification is independent."""
    return env_bool("ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION", True)


def object_relations_enabled():
    return env_bool("ENABLE_OBJECT_RELATIONS", True)


def group_llm_filter_enabled():
    return env_bool("ENABLE_GROUP_LLM_FILTER", True)


def room_llm_filter_enabled():
    return env_bool("ENABLE_ROOM_LLM_FILTER", True)


def instance_confirmation_enabled():
    """Master switch for instance checks, independent of group/room filtering."""
    if _raw('ENABLE_INSTANCE_CONFIRMATION') is not None:
        return env_bool('ENABLE_INSTANCE_CONFIRMATION', True)
    if _raw('ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL') is not None:
        warnings.warn('ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL is deprecated; '
                      'use ENABLE_INSTANCE_CONFIRMATION', DeprecationWarning, stacklevel=2)
        return env_bool('ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL', True)
    return True


def instance_confirmation_before_find_goal_enabled():
    """Compatibility alias for callers using the old master-switch name."""
    return instance_confirmation_enabled()


def instance_confirmation_threshold():
    return env_int('INSTANCE_CONFIRMATION_THRESHOLD', 1)


def verify_before_candidate_enabled():
    return instance_confirmation_enabled() and env_bool('VERIFY_BEFORE_CANDIDATE', True)


def instance_confirmation_mode():
    if "ENABLE_CONTRADICTION_CHECK" in os.environ:
        raise ValueError(
            'ENABLE_CONTRADICTION_CHECK is retired; use INSTANCE_CONFIRMATION_MODE='
            'positive_match (old false) or contradiction (old true)'
        )
    mode = (_raw("INSTANCE_CONFIRMATION_MODE") or "positive_match").lower()
    if mode not in {"positive_match", "contradiction"}:
        raise ValueError('INSTANCE_CONFIRMATION_MODE must be positive_match or contradiction')
    return mode


INSTANCE_VERIFICATION_PROMPTS = ('match', 'match_strict_no')


def instance_verification_prompt():
    """Wording of the positive_match yes/no/unknown rule; the yes wording is shared.

    match: the original prompt ("If evidence is inadequate, answer unknown.").
    match_strict_no: "no" only for a clear contradiction; unclear or not visible is "unknown".
    """
    name = (_raw('INSTANCE_VERIFICATION_PROMPT') or 'match').lower()
    if name not in INSTANCE_VERIFICATION_PROMPTS:
        raise ValueError('INSTANCE_VERIFICATION_PROMPT must be one of '
                         + ', '.join(INSTANCE_VERIFICATION_PROMPTS) + f', got {name!r}')
    return name


def instance_verification_scope():
    value = (_raw('INSTANCE_VERIFICATION_SCOPE') or 'instance').lower()
    if value not in {'instance', 'category'}:
        raise ValueError('INSTANCE_VERIFICATION_SCOPE must be instance or category')
    return value


def instance_rejection_policy():
    value = (_raw('INSTANCE_REJECTION_POLICY') or 'none').lower()
    if value not in {'none', 'no', 'non_yes'}:
        raise ValueError('INSTANCE_REJECTION_POLICY must be none, no or non_yes')
    return value


def instance_confirmation_count_unit():
    value = (_raw('INSTANCE_CONFIRMATION_COUNT_UNIT') or 'detection').lower()
    if value not in {'detection', 'frame'}:
        raise ValueError('INSTANCE_CONFIRMATION_COUNT_UNIT must be detection or frame')
    return value


def validate_instance_ablation():
    """Fail before a run rather than silently combining incompatible semantics."""
    scope, policy = instance_verification_scope(), instance_rejection_policy()
    if scope != 'instance' or policy != 'none':
        if not instance_confirmation_enabled() or instance_confirmation_mode() != 'positive_match':
            raise ValueError('Category/rejection ablations require confirmation and positive_match')
        if (verify_before_candidate_enabled() or instance_collect_only_enabled()
                or relative_evidence_enabled() or post_confirmation_contradiction_check_enabled()):
            raise ValueError('Category/rejection ablations require admission, collect-only, '
                             'relative evidence and post check disabled')
    return {'scope': scope, 'rejection_policy': policy, 'count_unit': instance_confirmation_count_unit(),
            'record_inputs': env_bool('INSTANCE_VERIFICATION_RECORD_INPUTS', False)}


def post_confirmation_contradiction_check_enabled():
    if not instance_confirmation_enabled():
        return False
    mode = instance_confirmation_mode()
    enabled = env_bool("ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK", False)
    if enabled and mode != "positive_match":
        raise ValueError(
            'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK requires '
            'INSTANCE_CONFIRMATION_MODE=positive_match and '
            'ENABLE_INSTANCE_CONFIRMATION=true'
        )
    return enabled


def instance_collect_only_enabled():
    """Data-collection mode for offline replay (实验计划.md S6); never a navigation method.

    Every lock the configured rule would make is logged as an ``[Intended Lock]``
    and its candidate is retired from approach; the agent never commits and keeps
    exploring until the step budget ends. Requires positive_match confirmation.
    """
    enabled = env_bool('INSTANCE_COLLECT_ONLY', False)
    if enabled and (not instance_confirmation_enabled() or instance_confirmation_mode() != 'positive_match'):
        raise ValueError('INSTANCE_COLLECT_ONLY requires ENABLE_INSTANCE_CONFIRMATION=true '
                         'and INSTANCE_CONFIRMATION_MODE=positive_match')
    if enabled and post_confirmation_contradiction_check_enabled():
        raise ValueError('INSTANCE_COLLECT_ONLY cannot be combined with '
                         'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK')
    return enabled


def preserve_positive_evidence_on_recovery_enabled():
    if not instance_confirmation_enabled():
        return False
    enabled = env_bool("PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY", False)
    if enabled and instance_confirmation_mode() != "positive_match":
        raise ValueError(
            'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY requires '
            'INSTANCE_CONFIRMATION_MODE=positive_match and '
            'ENABLE_INSTANCE_CONFIRMATION=true'
        )
    return enabled


RELATIVE_EVIDENCE_TOGGLES = (
    'ENABLE_RELATIVE_EVIDENCE', 'RELATIVE_EVIDENCE_CALIBRATION',
    'RELATIVE_EVIDENCE_USE_U', 'RELATIVE_EVIDENCE_EARLY_STOP',
    'RELATIVE_EVIDENCE_PAIRWISE', 'RELATIVE_EVIDENCE_DROP_REJECTED',
)


def relative_evidence_enabled():
    return env_bool('ENABLE_RELATIVE_EVIDENCE', False)


def relative_evidence_settings():
    return {
        'calibration': _raw('RELATIVE_EVIDENCE_CALIBRATION') or str(
            Path(__file__).resolve().parent / 'relative_evidence_s6.json'),
        'use_u': env_bool('RELATIVE_EVIDENCE_USE_U', True),
        'early_stop': env_bool('RELATIVE_EVIDENCE_EARLY_STOP', True),
        'pairwise': env_bool('RELATIVE_EVIDENCE_PAIRWISE', True),
        'drop_rejected': env_bool('RELATIVE_EVIDENCE_DROP_REJECTED', True),
    }


def validate_relative_evidence(*, rag_mode=None, dataset=None, backend=None):
    """Validate the studied S6 configuration without silently changing other switches."""
    if not relative_evidence_enabled():
        return None
    required = {
        'ENABLE_INSTANCE_CONFIRMATION=true': instance_confirmation_enabled(),
        'INSTANCE_CONFIRMATION_MODE=positive_match': instance_confirmation_mode() == 'positive_match',
        'VERIFY_BEFORE_CANDIDATE=false': not verify_before_candidate_enabled(),
        'INSTANCE_CONFIRMATION_THRESHOLD=1': instance_confirmation_threshold() == 1,
        'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK=false': not post_confirmation_contradiction_check_enabled(),
        'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY=false': not preserve_positive_evidence_on_recovery_enabled(),
        'INSTANCE_COLLECT_ONLY=false': not instance_collect_only_enabled(),
        'RAG_MODE=0': str(rag_mode if rag_mode is not None else os.environ.get('RAG_MODE', '0')) == '0',
    }
    if dataset is not None:
        required['dataset=instancenav'] = dataset == 'instancenav'
    settings = relative_evidence_settings()
    if backend is not None and settings['pairwise']:
        required['multi-image backend=openai'] = backend == 'openai'
    conflicts = [name for name, valid in required.items() if not valid]
    if conflicts:
        raise ValueError('Relative evidence requires ' + ', '.join(conflicts))
    return settings


def relative_evidence_run_header(**validation):
    settings = validate_relative_evidence(**validation)
    if settings is None:
        return {'enabled': False}
    from utils.relative_evidence import Calibration
    calibration = Calibration(settings['calibration'])
    return {'enabled': True, **settings, **calibration.metadata()}
