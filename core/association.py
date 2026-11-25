"""Data association: cost construction and gated Hungarian matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .motion import iou_batch

INFEASIBLE = 1e6


def solve(cost: np.ndarray, valid: np.ndarray = None):
    """Hungarian assignment that respects infeasible pairs."""
    if cost.size == 0:
        return []
    feasible = np.isfinite(cost)
    if valid is not None:
        feasible &= valid
    if not feasible.any():
        return []

    safe = np.where(feasible, cost, INFEASIBLE)
    rows, cols = linear_sum_assignment(safe)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if feasible[r, c]]


def iou_cost(track_boxes, det_boxes, w_iou: float, w_app: float,
             similarities: np.ndarray = None, gate_iou: float = 0.10,
             gate_app: float = 0.50):
    """Build an association cost matrix and its feasibility mask.

    ``cost = w_iou * (1 - IoU) + w_app * (1 - appearance_similarity)``
    """
    tracks = np.asarray(track_boxes, dtype=float).reshape(-1, 4)
    dets = np.asarray(det_boxes, dtype=float).reshape(-1, 4)

    iou = iou_batch(tracks, dets)
    if similarities is None:
        similarities = np.zeros_like(iou)
    else:
        similarities = np.asarray(similarities, dtype=float).reshape(iou.shape)

    cost = w_iou * (1.0 - iou) + w_app * (1.0 - similarities)
    valid = (iou >= gate_iou) | (similarities >= gate_app)
    cost = np.where(valid, cost, INFEASIBLE)
    return cost, iou, similarities, valid


def greedy_solve(cost: np.ndarray, valid: np.ndarray):
    """Fallback association: take the globally cheapest *feasible* pairs.

    The feasibility mask is authoritative.  An earlier version checked the
    masked cost for ``inf`` and therefore happily assigned impossible pairs
    (a track could be matched to a detection 500 px away), so this function now
    only ever walks valid entries.
    """
    if cost.size == 0:
        return []
    flat_cost = np.where(valid, cost, np.inf).ravel()
    if not np.isfinite(flat_cost).any():
        return []

    pairs = []
    used_rows, used_cols = set(), set()
    for index in np.argsort(flat_cost):
        value = float(flat_cost[index])
        if not np.isfinite(value):
            break
        row, col = divmod(int(index), cost.shape[1])
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        pairs.append((row, col))
    return pairs
