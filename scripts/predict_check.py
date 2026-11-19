"""Missing-edge check: hide 10% of the direct edges, train on the rest (neg_filter told), rank the hidden parents.
python scripts/predict_check.py <data prefix> [--init text] [--seeds 0 1]"""
import argparse, json, random, time
from regd import RegD
from regd.benchmarks import load_split, load_vocab

ap = argparse.ArgumentParser()
ap.add_argument("prefix"); ap.add_argument("--init", default="random"); ap.add_argument("--dim", type=int, default=5)
ap.add_argument("--preset", default="strong-bce"); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
ap.add_argument("--filter", default="told"); ap.add_argument("--out", default="predict.jsonl")
a = ap.parse_args()
voc = load_vocab(a.prefix)
edges = [(voc[c], voc[p]) for c, p in load_split(a.prefix)["train"]]
texts = {n: n.split(".")[0].replace("_", " ") for e in edges for n in e}
for seed in a.seeds:
    rng = random.Random(seed); e = edges[:]; rng.shuffle(e)
    k = len(e) // 10
    held, train = e[:k], e[k:]
    t0 = time.time()
    m = RegD(dim=a.dim, preset=a.preset, neg_filter=a.filter, init=a.init, seed=seed)
    m.fit(train, nodes=sorted(texts), texts=texts, verbose=False)
    res = {"data": a.prefix.split("/")[-1], "init": a.init, "preset": a.preset, "filter": a.filter, "dim": a.dim,
           "seed": seed, "held_out": m.rank(held), "train_edges": m.rank(train[:500]), "containment": m.containment(),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    open(a.out, "a").write(json.dumps(res) + "\n")
