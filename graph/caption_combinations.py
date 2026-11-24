"""Score group captions without materializing the Cartesian product for top-k."""
from heapq import heappop, heappush, nlargest
from itertools import product
from operator import itemgetter


def _probability(values):
    # Keep the legacy left-to-right floating-point operations, including underflow.
    score = 1.0
    for p in values:
        score *= p
    return score


def _exhaustive(captions, probs):
    for labels, values in zip(product(*captions), product(*probs)):
        yield labels, _probability(values)


def scored_captions(captions, probs, top_k):
    """Yield (captions, score) for the caller's stable descending sort and slice."""
    exhaustive = _exhaustive(captions, probs)
    if top_k is None or top_k < 0 or not all(
        type(p) in (int, float) and 0 <= p <= 1 for axis in probs for p in axis
    ):
        # Preserve legacy arithmetic/sort semantics for nonstandard numeric types,
        # non-probabilities (e.g. NaN), full enumerations and negative slices.
        yield from exhaustive
        return
    if any(a < b for axis in probs for a, b in zip(axis, axis[1:])):
        # Do not reorder an input axis: tied scores must retain product order.
        yield from nlargest(top_k, exhaustive, key=itemgetter(1))
        return

    # A coordinate increment cannot increase a nonnegative product, and makes
    # the index tuple lexicographically larger. Thus each parent precedes its
    # children under (-score, index), even for rounded ties or all-zero scores.
    # Popping the frontier in that order gives EXACTLY stable-sort(product)[:k].
    start = (0,) * len(probs)
    if not top_k or not all(probs):
        return
    heap = [(-_probability(axis[0] for axis in probs), start)]
    visited = {start}
    for rank in range(top_k):
        if not heap:
            break
        neg_score, indices = heappop(heap)
        yield tuple(axis[i] for axis, i in zip(captions, indices)), -neg_score
        if rank + 1 == top_k:
            break
        for dim, i in enumerate(indices):
            if i + 1 == len(probs[dim]):
                continue
            child = indices[:dim] + (i + 1,) + indices[dim + 1:]
            if child not in visited:
                visited.add(child)
                score = _probability(axis[j] for axis, j in zip(probs, child))
                heappush(heap, (-score, child))
