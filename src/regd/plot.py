"""Picture of the regions in two of the dimensions (needs matplotlib)."""
from __future__ import annotations

import numpy as np


def draw(model, path=None, dims=(0, 1), labels=12, ax=None):
    """Draw every node's box / ball projected on two dimensions; the `labels` largest regions are named.
    Returns the matplotlib axes; saves a figure when `path` is given."""
    import matplotlib
    if path is not None and ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    r = model.regions()
    c = r["center"][:, list(dims)]
    box = "offset" in r
    size = r["offset"][:, list(dims)] if box else np.repeat(r["radius"][:, None], 2, 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    order = np.argsort(-size.prod(1))
    for rank, i in enumerate(order):
        kw = dict(fill=False, linewidth=0.6, edgecolor=plt.cm.viridis(rank / max(1, len(order) - 1)), alpha=0.8)
        ax.add_patch(Rectangle(c[i] - size[i], *(2 * size[i]), **kw) if box else Circle(c[i], size[i, 0], **kw))
    for i in order[:labels]:
        ax.annotate(str(r["nodes"][i]), c[i] + [-size[i, 0], size[i, 1]], fontsize=7, va="bottom")
    lo, hi = (c - size).min(0), (c + size).max(0)
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal")
    ax.set_xlabel(f"dimension {dims[0]}")
    ax.set_ylabel(f"dimension {dims[1]}")
    if path is not None:
        ax.figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(ax.figure)
    return ax
