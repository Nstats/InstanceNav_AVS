"""Collect-only mode (INSTANCE_COLLECT_ONLY, 实验计划.md S6).

The navigator runs exactly as the baseline until the moment it would lock a
candidate. Instead of locking, it logs an ``[Intended Lock]`` record, retires that
candidate track from approach and locking (it stays in the evidence store and
keeps receiving verification answers and votes whenever it is detected again)
and keeps exploring. The episode therefore ends at the step budget, and the log
holds every candidate, every verification answer and periodic exploration state
for offline replay of stopping rules.

Records added (all JSON, one line each):
- ``[Intended Lock]``: the lock the baseline rule would have made, with the
  state of every candidate track at that moment;
- ``[Collect Frame]``: every kept box of a verified frame with its track id and
  agent-side position estimate (far boxes included);
- ``[Collect Snapshot]``: explored area, frontier clusters with a geometric
  information-gain proxy (U-a), and the room map summary, every SNAPSHOT_EVERY
  policy steps and at every intended lock;
- ``[Landmark Utility]``: PSG-Nav's per-landmark win rate, prior and utility
  each time it chooses an exploration landmark.

Nothing here reads ground truth. With the switch off none of this code runs.
"""
import numpy as np

from utils.log_utils import log_event

# Exploration snapshot cadence (policy steps); also emitted at every intended lock.
SNAPSHOT_EVERY = 10
# Detection range used by the frontier information-gain proxy (meters).
GAIN_RADIUS_M = 5.0


def guard_action(agent, action):
    """Only called in collection mode: recovery exhaustion must not issue STOP."""
    if action != 0:
        return action
    log_event('Collect Suppressed Stop', episode=getattr(agent, 'count_episodes', None),
              step=agent.total_steps, reason='planner_returned_stop', replacement_action=3,
              found_goal=bool(agent.found_goal))
    if agent.found_goal:
        agent.reset_goal_detection(reason='collect_suppressed_stop')
    # An ordinary 30-degree turn consumes one atomic action and obtains a new view.
    return 3


def _rounded(values, digits=3):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr.round(digits).tolist()


def _known_masks(agent):
    free = agent.fbe_free_map[0, 0].detach().cpu().numpy() > 0
    obstacle = agent.full_map[0, 0].detach().cpu().numpy() > 0
    return free, obstacle


def map_coverage(agent):
    """Known-space statistics from the agent's own maps (cells of agent.resolution cm)."""
    free, obstacle = _known_masks(agent)
    cell_m2 = (agent.resolution / 100.0) ** 2
    return {'known_m2': round(float((free | obstacle).sum()) * cell_m2, 2),
            'free_m2': round(float(free.sum()) * cell_m2, 2)}


def frontier_summary(agent, max_clusters=12):
    """Frontier cells, their clusters and a geometric information-gain proxy (U-a).

    Each cluster's gain is the number of unknown cells within GAIN_RADIUS_M of its
    centre; gain_union_m2 counts the union over all clusters, so overlapping views
    of the same unknown area are not double counted.
    """
    from utils.utils_frontiers import calculate_frontiers
    _, locations, count = calculate_frontiers(agent.full_map, agent.fbe_free_map)
    if locations is None or count == 0:
        return {'frontier_cells': 0, 'n_clusters': 0, 'clusters': [], 'gain_union_m2': 0.0}
    cells = locations.detach().cpu().numpy().astype(int)
    free, obstacle = _known_masks(agent)
    unknown = ~(free | obstacle)
    size = unknown.shape[0]
    cell_m2 = (agent.resolution / 100.0) ** 2
    radius = max(1, int(GAIN_RADIUS_M * 100 / agent.resolution))
    # Coarse clustering on a grid of the gain radius: cheap and deterministic.
    clusters = {}
    for key, cell in zip(map(tuple, cells // radius), cells):
        clusters.setdefault(key, []).append(cell)
    union = np.zeros_like(unknown)
    out = []
    agent_rc = getattr(agent, 'agent_map_pos', None)
    for members in sorted(clusters.values(), key=len, reverse=True):
        members = np.asarray(members)
        r, c = members.mean(axis=0).astype(int)
        r0, r1 = max(0, r - radius), min(size, r + radius + 1)
        c0, c1 = max(0, c - radius), min(size, c + radius + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        window = unknown[r0:r1, c0:c1] & ((yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2)
        union[r0:r1, c0:c1] |= window
        if len(out) < max_clusters:
            dist = None
            if agent_rc is not None:
                # Frontier rows follow fbe_free_map; agent_map_pos uses the flipped row axis (see fbe()).
                dist = round(float(np.hypot(size - r - agent_rc[0], c - agent_rc[1]))
                             * agent.resolution / 100.0, 2)
            out.append({'rc': [int(r), int(c)], 'cells': int(len(members)),
                        'gain_m2': round(float(window.sum()) * cell_m2, 2), 'dist_m': dist})
    return {'frontier_cells': int(count), 'n_clusters': len(clusters), 'clusters': out,
            'gain_union_m2': round(float(union.sum()) * cell_m2, 2)}


def room_summary(agent, min_cells=20):
    """Mapped area per room channel (agent-side room map), rooms with >= min_cells only."""
    from configs.dataset_categories import ROOM_CATEGORIES
    rm = agent.room_map[0].detach().cpu().numpy() > 0
    counts = rm.reshape(rm.shape[0], -1).sum(axis=1)
    cell_m2 = (agent.resolution / 100.0) ** 2
    return {name: round(float(n) * cell_m2, 2)
            for name, n in zip(ROOM_CATEGORIES, counts) if n >= min_cells}


def candidate_summary(store, retired_ids):
    """Every candidate track: position, verification history, and retirement state."""
    if store is None:
        return []
    return [{'track_id': t.track_id,
             'node_id': (str(t.node_id) if t.node_id is not None else None),
             'gps': _rounded(t.gps), 'last_step': t.step, 'votes': len(t.votes),
             'positive_steps': len(t.positive_steps), 'vetoed': bool(t.vetoed),
             'retired': t.track_id in retired_ids}
            for t in store.tracks]


def snapshot(agent, *, reason, store=None, retired_ids=()):
    """One [Collect Snapshot] record of agent-side exploration state."""
    rec = {'episode': getattr(agent, 'count_episodes', None),
           'step': getattr(agent, 'total_steps', None), 'reason': reason}
    for key, fn in (('coverage', map_coverage), ('frontiers', frontier_summary), ('rooms', room_summary)):
        try:
            rec[key] = fn(agent)
        except Exception as error:  # diagnostics must never break navigation
            rec[key] = {'error': f'{type(error).__name__}: {error}'[:200]}
    rec['n_candidates'] = len(store.tracks) if store is not None else 0
    rec['n_retired'] = len(retired_ids)
    log_event('Collect Snapshot', **rec)


def log_intended_lock(agent, *, path, track, store, retired_ids):
    log_event('Intended Lock', episode=getattr(agent, 'count_episodes', None),
              step=getattr(agent, 'total_steps', None), path=path,
              track_id=track.track_id, goal_gps=_rounded(track.gps),
              distance_m=_rounded(track.current.get('distance')),
              box=[int(v) for v in track.current.get('box') or []] or None,
              votes=len(track.votes), n_retired_before=len(retired_ids),
              candidates=candidate_summary(store, retired_ids))


def log_frame(agent, *, candidates, tracks, kept, confirmed, positive, retired_ids):
    """Every kept box of one verified frame, including far boxes never re-emitted as [Detection]."""
    boxes = []
    for index in kept:
        candidate, track = candidates[index], tracks[index]
        boxes.append({'box_index': index, 'track_id': track.track_id if track is not None else None,
                      'box': [int(v) for v in candidate['box']], 'gps': _rounded(candidate['gps']),
                      'distance_m': _rounded(candidate['distance']),
                      'confirmed': index in confirmed, 'positive': index in positive,
                      'retired': track is not None and track.track_id in retired_ids})
    log_event('Collect Frame', episode=getattr(agent, 'count_episodes', None),
              step=getattr(agent, 'total_steps', None), boxes=boxes)


def log_landmark_utilities(agent, landmarks, utilities, best_index):
    log_event('Landmark Utility', episode=getattr(agent, 'count_episodes', None),
              step=getattr(agent, 'total_steps', None), best=int(best_index),
              landmarks=[{'original_index': lm.get('original_index'),
                          'position': _rounded(lm.get('position'), 1),
                          'win_count': lm.get('win_count'), 'total_comparisons': lm.get('total_comparisons'),
                          'prior': _rounded(lm.get('prior_probability')),
                          'utility': _rounded(u),
                          'room': (lm.get('geometry') or {}).get('room_name'),
                          'exploration_pct': _rounded((lm.get('geometry') or {}).get('exploration_pct')),
                          'info_gain': _rounded((lm.get('geometry') or {}).get('info_gain'))}
                         for lm, u in zip(landmarks, utilities)])
