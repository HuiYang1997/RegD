"""Train a model on one of the paper's data sets with the node names as names, and save it:
python scripts/pretrain.py <data prefix> --dim 5 --region box --out mammal_box_d5.pt [--preset strong-bce --set lr=0.005 ...]
Prints one JSON line with the test result of the saved model."""
import argparse, json, time
from regd import RegD
from regd.benchmarks import load_split, load_vocab

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--dim", type=int, default=5)
ap.add_argument("--region", default="box")
ap.add_argument("--preset", default="strong-bce")
ap.add_argument("--seed", type=int, default=43)
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
opts = {}
for kv in a.set:
    k, v = kv.split("=", 1)
    try: v = json.loads(v)
    except ValueError: pass
    opts[k] = v
voc = load_vocab(a.prefix)
d = {k: [(voc[c], voc[p]) for c, p in v] for k, v in load_split(a.prefix).items()}
t0 = time.time()
m = RegD(dim=a.dim, region=a.region, preset=a.preset, seed=a.seed, **opts)
m.fit(d["train"], valid=(d["valid"], d["valid_neg"]), nodes=sorted(voc.values()), verbose=False)
m.save(a.out)
m = RegD.load(a.out)
print(json.dumps({"file": a.out, "data": a.prefix.split("/")[-1], "dim": a.dim, "region": a.region, "preset": a.preset,
                  "seed": a.seed, **opts, "nodes": m.hierarchy.N, "edges": len(m.hierarchy.edges), "tau": m.tau,
                  "test": m.evaluate(d["test"], d["test_neg"]), "containment": m.containment(),
                  "seconds": round(time.time() - t0, 1)}), flush=True)
