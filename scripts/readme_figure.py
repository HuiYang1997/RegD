"""The figure at the top of the README: the regions of a few species and of all their ancestors, from a model
trained in 2 dimensions.
python scripts/pretrain.py paper/data_utils/data/maxn/mammal_closure.tsv --dim 2 --region ball \
    --set lam=1.0 p=2 lr=0.01 batch_size=32 epochs=400 device=cpu --out mammal_ball_d2.pt
python scripts/readme_figure.py mammal_ball_d2.pt docs/regions.png bengal_tiger.n.01 toy_poodle.n.01 ..."""
import collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from regd import RegD

path, out, leaves = sys.argv[1], sys.argv[2], sys.argv[3:]
m = RegD.load(path)
h, r = m.hierarchy, m.regions()
box = "offset" in r
parents = collections.defaultdict(list)
for p, c in h.edges:
    parents[int(c)].append(int(p))


def chain(i):
    """i and its ancestors, the most general first."""
    seen, todo = [], [i]
    while todo:
        j = todo.pop()
        if j not in seen:
            seen.append(j)
            todo += parents[j]
    return sorted(seen, key=lambda j: -float(np.prod(r["offset"][j]) if box else r["radius"][j]))


short = lambda i: str(h.nodes[i]).split(".")[0].replace("_", " ")
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
chains = [chain(int(h.index[n])) for n in leaves]
count = collections.Counter(i for ch in chains for i in ch)
fig, ax = plt.subplots(figsize=(10.5, 7.2))
lo, hi, drawn = [], [], set()
for k, ch in enumerate(chains):
    for i in ch:
        if i in drawn:
            continue
        drawn.add(i)
        col = "#555555" if count[i] > 1 else colors[k % len(colors)]
        c = r["center"][i]
        size = r["offset"][i] if box else np.repeat(r["radius"][i], 2)
        shape = (lambda **kw: Rectangle(c - size, *(2 * size), **kw)) if box else (lambda **kw: Circle(c, size[0], **kw))
        ax.add_patch(shape(facecolor=col, alpha=0.07, edgecolor="none"))
        ax.add_patch(shape(fill=False, edgecolor=col, linewidth=1.5 if count[i] == 1 else 1.1))
        lo.append(c - size); hi.append(c + size)
        if i == ch[-1]:                              # the species itself: name in the middle of its region
            ax.annotate(short(i), c, ha="center", va="center", fontsize=8.5, color=col, weight="bold")
        elif count[i] > 1:                           # a shared ancestor: name at the top of its region
            ax.annotate(short(i), c + [0, size[1]], xytext=(0, -4), textcoords="offset points", ha="center",
                        va="top", fontsize=9, color=col)
lo, hi = np.min(lo, 0), np.max(hi, 0)
pad = 0.03 * (hi - lo)
ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
y = 1.0
ax.text(1.03, y, "from the outside in:", transform=ax.transAxes, fontsize=9, color="#555555", va="top")
y -= 0.015
for k, ch in enumerate(chains):
    y -= 0.03
    for i in ch:
        col = "#555555" if count[i] > 1 else colors[k % len(colors)]
        ax.text(1.03, y, short(i), transform=ax.transAxes, fontsize=9, color=col, va="top",
                weight="bold" if i == ch[-1] else "normal")
        y -= 0.036
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
pairs = [(a, b) for ch in chains for x, a in enumerate(ch) for b in ch[:x]]
pairs = sorted(set(pairs))
inside = sum(m.score(h.nodes[a], h.nodes[b]) <= 0 for a, b in pairs)
print(f"{len(drawn)} regions; {inside} of {len(pairs)} (descendant, ancestor) pairs among them are contained")
