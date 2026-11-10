"""Expand PSG's 60-degree scan into ordinary Habitat 30-degree turns."""


def expand_atomic_actions(action, enabled=False):
    if enabled and action.get('action') == 6:
        return [{'action': 3}, {'action': 3}]
    return [action]
