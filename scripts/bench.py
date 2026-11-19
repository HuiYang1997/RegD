"""Benchmark on the paper's splits:  python scripts/bench.py <data prefix> [--dim 5 --region box --preset strong
--seeds 43 0 --set lam=0.5 ...]   Appends one JSON line per run to --out."""
import argparse, json, time
from regd import RegD
from regd.benchmarks import load_split, load_vocab

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--dim", type=int, default=5)
ap.add_argument("--region", default="box")
ap.add_argument("--preset", default="strong")
ap.add_argument("--seeds", type=int, nargs="*", default=[43])
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--names", action="store_true", help="use the vocab names as node names (needed for init=text)")
ap.add_argument("--rank_cap", type=int, default=2000)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="bench.jsonl")
a = ap.parse_args()
opts = {}
for kv in a.set:
    k, v = kv.split("=", 1)
    try: v = json.loads(v)
    except ValueError: pass
    opts[k] = v
d = load_split(a.prefix)
if a.names:
    voc = load_vocab(a.prefix)
    name = lambda x: voc[x]
    d = {k: [(name(c), name(p)) for c, p in v] for k, v in d.items()}
texts = {n: n.split(".")[0].replace("_", " ") for v in d.values() for pr in v for n in pr} if a.names else None
for seed in a.seeds:
    t0 = time.time()
    m = RegD(dim=a.dim, region=a.region, preset=a.preset, seed=seed, **opts)
    m.fit(d["train"], valid=(d["valid"], d["valid_neg"]), nodes=[n for k in ("test", "test_neg") for pr in d[k] for n in pr],
          verbose=False, texts=texts)
    res = {"tag": a.tag, "data": a.prefix.split("/")[-1], "dim": a.dim, "region": a.region, "preset": a.preset, "seed": seed, **opts,
           "val_f1": max(h["val_f1"] for h in m.history), "tau": m.tau, "test": m.evaluate(d["test"], d["test_neg"]),
           "rank": m.rank(d["test"][:a.rank_cap]), "containment": m.containment(),
           "trans_viol": m.transitivity_violations(), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
