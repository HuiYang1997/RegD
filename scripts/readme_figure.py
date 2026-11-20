"""The figure at the top of the README, in the style of the figure of the paper: every node of a ball model as a
point (sum of the centre coordinates, radius), the training edges as lines, a few classes named.
python scripts/pretrain.py paper/data_utils/data/maxn/mammal_closure.tsv --dim 2 --region ball \
    --set lam=1.0 p=2 lr=0.01 batch_size=32 epochs=400 device=cpu --out mammal_ball_d2.pt
python scripts/readme_figure.py mammal_ball_d2.pt docs/regions.png placental.n.01 carnivore.n.01 ungulate.n.01@-7,7 ..."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from regd import RegD

path, out, named = sys.argv[1], sys.argv[2], sys.argv[3:]
m = RegD.load(path)
h, r = m.hierarchy, m.regions()
x, y = r["center"].sum(1), r["radius"]
fig, ax = plt.subplots(figsize=(10, 8))
ax.add_collection(LineCollection([[(x[p], y[p]), (x[c], y[c])] for p, c in h.edges], colors="black",
                                 linewidths=0.5, alpha=0.25))
ax.scatter(x, y, s=22, c="blue", alpha=0.6)
for n in named:                                     # "node" or "node@dx,dy": where the name goes, in points
    n, _, off = n.partition("@")
    dx, dy = map(float, off.split(",")) if off else (7, 7)
    i = int(h.index[n])
    ax.scatter(x[i], y[i], s=70, c="red", edgecolors="black", linewidths=1.2, zorder=3)
    ax.annotate(str(n).split(".")[0], (x[i], y[i]), xytext=(dx, dy), textcoords="offset points", fontsize=10,
                ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom")
ax.set_xlabel("Center Sum")
ax.set_ylabel("Size")
ax.grid(alpha=0.3)
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
