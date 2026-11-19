"""Threshold selection and metrics."""
from __future__ import annotations

import numpy as np


def best_threshold(scores, labels, tau_max=None):
    """Threshold tau maximising F1 of the rule 'score <= tau' (exact search over the observed scores).

    tau_max = 0 restricts the search to tau <= 0, the range in which the transitive rules are transitive.
    Returns (tau, f1).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if not len(s) or y.sum() == 0:
        return (0.0 if tau_max is None else tau_max), 0.0
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of every distinct score
    if tau_max is not None:
        last = last[s[last] <= tau_max]
        if not len(last):
            return float(tau_max), 0.0
    f1 = 2 * tp[last] / (tp[last] + fp[last] + y.sum())
    j = int(np.argmax(f1))
    i = last[j]
    tau = s[i] if i + 1 >= len(s) else 0.5 * (s[i] + s[i + 1])  # midway to the next score
    if tau_max is not None:
        tau = min(tau, tau_max)
    return float(tau), float(f1[j])


def f1_at(scores, labels, tau):
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    pred = s <= tau
    tp = float((pred & y).sum())
    precision = tp / max(float(pred.sum()), 1.0)
    recall = tp / max(float(y.sum()), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_rank_metrics(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return {"mrr": float((1 / r).mean()), "hits@1": float((r <= 1).mean()), "hits@3": float((r <= 3).mean()),
            "hits@10": float((r <= 10).mean()), "mean_rank": float(r.mean()), "n": int(len(r))}
