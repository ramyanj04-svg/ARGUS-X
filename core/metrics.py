"""Identity-centric evaluation (MOTA / IDF1 / ID switches / fragmentation).

These are the standard CLEAR-MOT and Identity metrics, computed without any
external dependency.  They are used by :mod:`core.benchmark` on a synthetic
sequence where the ground truth is known exactly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .motion import iou_batch


def _greedy_match(gt_boxes: np.ndarray, pred_boxes: np.ndarray, threshold: float = 0.5):
    """Greedy IoU matching, highest overlap first."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    ious = iou_batch(gt_boxes, pred_boxes)
    pairs = []
    used_g, used_p = set(), set()
    order = np.argsort(ious.ravel())[::-1]
    for index in order:
        g, p = divmod(int(index), ious.shape[1])
        if ious[g, p] < threshold:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        pairs.append((g, p, float(ious[g, p])))
    return pairs


def mot_metrics(gt: Dict[int, List[Tuple[str, list, bool]]],
                pred: Dict[int, List[Tuple[str, list]]],
                iou_threshold: float = 0.5) -> dict:
    """Compute MOTA, IDF1, ID switches and fragmentation.

    ``gt`` maps frame -> [(track_label, bbox, visible)] and ``pred`` maps
    frame -> [(track_label, bbox)].  Invisible ground-truth objects (fully
    occluded) are excluded from false negatives, exactly like a real benchmark
    annotation would flag them.
    """
    frames = sorted(set(gt.keys()) | set(pred.keys()))
    total_gt = 0
    false_negatives = 0
    false_positives = 0
    id_switches = 0
    matches = 0
    iou_sum = 0.0

    last_match: Dict[str, str] = {}
    gt_matched_in_frame: Dict[str, bool] = {}
    gt_match_history: Dict[str, List[bool]] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}
    gt_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}

    for frame in frames:
        gt_items = [(label, box) for label, box, visible in gt.get(frame, []) if visible]
        pred_items = pred.get(frame, [])
        for label, _ in gt_items:
            gt_counts[label] = gt_counts.get(label, 0) + 1
        for label, _ in pred_items:
            pred_counts[label] = pred_counts.get(label, 0) + 1

        total_gt += len(gt_items)
        gboxes = np.array([box for _, box in gt_items], dtype=float)
        pboxes = np.array([box for _, box in pred_items], dtype=float)
        pairs = _greedy_match(gboxes, pboxes, iou_threshold) if len(gt_items) and len(pred_items) else []

        matched_gt = set()
        matched_pred = set()
        for g, p, iou in pairs:
            gt_label = gt_items[g][0]
            pred_label = pred_items[p][0]
            matched_gt.add(g)
            matched_pred.add(p)
            matches += 1
            iou_sum += iou
            pair_counts[(gt_label, pred_label)] = pair_counts.get((gt_label, pred_label), 0) + 1
            previous = last_match.get(gt_label)
            if previous is not None and previous != pred_label:
                id_switches += 1
            last_match[gt_label] = pred_label

        false_negatives += len(gt_items) - len(matched_gt)
        false_positives += len(pred_items) - len(matched_pred)

        for g, (label, _) in enumerate(gt_items):
            hit = g in matched_gt
            history = gt_match_history.setdefault(label, [])
            if hit and history and not history[-1]:
                gt_matched_in_frame[label] = True
            elif hit:
                gt_matched_in_frame.setdefault(label, False)
            history.append(hit)

    # ---- IDF1 -----------------------------------------------------------------
    if pair_counts:
        gt_labels = sorted(gt_counts)
        pred_labels = sorted(pred_counts)
        matrix = np.zeros((len(gt_labels), len(pred_labels)))
        for (gt_label, pred_label), count in pair_counts.items():
            matrix[gt_labels.index(gt_label), pred_labels.index(pred_label)] = count
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-matrix)
        idtp = float(matrix[rows, cols].sum())
    else:
        idtp = 0.0

    idfn = sum(gt_counts.values()) - idtp
    idfp = sum(pred_counts.values()) - idtp
    idf1 = (2 * idtp / (2 * idtp + idfp + idfn)) if (2 * idtp + idfp + idfn) > 0 else 0.0

    # ---- fragmentation --------------------------------------------------------
    fragments = 0
    for label, history in gt_match_history.items():
        transitions = sum(1 for a, b in zip(history, history[1:]) if a and not b)
        fragments += transitions

    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(total_gt, 1)
    precision = matches / max(matches + false_positives, 1)
    recall = matches / max(total_gt, 1)

    return {
        "frames": len(frames),
        "gt_objects": int(total_gt),
        "matches": int(matches),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "id_switches": int(id_switches),
        "fragments": int(fragments),
        "idf1": round(float(idf1), 4),
        "mota": round(float(mota), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "motp": round(float(iou_sum / matches), 4) if matches else 0.0,
        "unique_predictions": len(pred_counts),
        "unique_ground_truth": len(gt_counts),
    }
