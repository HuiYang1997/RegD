"""Command line:  regd train | query | eval | info | export | plot | options"""
from __future__ import annotations

import argparse
import json

from .graph import read_edges
from .model import PRESETS, RegD, options_table


def _options(pairs):
    opts = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        opts[k] = v
    return opts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="regd", description="Region embeddings of hierarchies (RegD)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on a file of 'child<TAB>parent' lines")
    t.add_argument("edges")
    t.add_argument("--out", required=True, help="where to save the model")
    t.add_argument("--dim", type=int, default=5)
    t.add_argument("--region", default="box", choices=["box", "ball"])
    t.add_argument("--preset", default="strong", choices=sorted(PRESETS))
    t.add_argument("--parent-first", action="store_true", help="lines are 'parent<TAB>child'")
    t.add_argument("--valid", nargs=2, metavar=("POS", "NEG"), help="validation pair files (same column order)")
    t.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="other options (see 'regd options'), e.g. --set lam=0 epochs=200 init=text  or  lam_set=[0,0.5,1]")
    t.add_argument("--quiet", action="store_true")

    q = sub.add_parser("query", help="ancestors / descendants of a node, or the score of a pair")
    q.add_argument("model")
    q.add_argument("node")
    q.add_argument("other", nargs="?", help="if given: score and decision of 'node is-a other'")
    q.add_argument("--descendants", action="store_true")
    q.add_argument("-k", type=int, default=None, help="the k best instead of all accepted")

    e = sub.add_parser("eval", help="precision / recall / F1 on pair files, and ranking metrics on the positives")
    e.add_argument("model")
    e.add_argument("pos")
    e.add_argument("neg")
    e.add_argument("--parent-first", action="store_true")

    i = sub.add_parser("info", help="size, options and validation result of a saved model")
    i.add_argument("model")

    x = sub.add_parser("export", help="write the regions to a .tsv or .npz file")
    x.add_argument("model")
    x.add_argument("out")

    p = sub.add_parser("plot", help="draw the regions in two dimensions (needs matplotlib)")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--dims", type=int, nargs=2, default=[0, 1])
    p.add_argument("--labels", type=int, default=12)

    sub.add_parser("options", help="list every option with its default, choices and meaning")

    a = ap.parse_args(argv)
    if a.cmd == "options":
        print(options_table())
        print("\npresets:")
        for name, o in PRESETS.items():
            print(f"  {name:<11} " + "  ".join(f"{k}={v}" for k, v in o.items()))
        return
    if a.cmd == "train":
        valid = [read_edges(f, a.parent_first) for f in a.valid] if a.valid else None
        m = RegD(dim=a.dim, region=a.region, preset=a.preset, **_options(a.set))
        m.fit(a.edges, valid=valid, parent_first=a.parent_first, verbose=not a.quiet)
        m.save(a.out)
        print(json.dumps({**m.summary(), "saved": a.out}))
        return
    m = RegD.load(a.model)
    if a.cmd == "query":
        if a.other:
            print(json.dumps({"score": m.score(a.node, a.other), "tau": m.tau, "is_a": m.is_a(a.node, a.other)}))
        else:
            for name, s in (m.descendants if a.descendants else m.ancestors)(a.node, a.k):
                print(f"{s:.4f}\t{name}")
    elif a.cmd == "eval":
        pos, neg = read_edges(a.pos, a.parent_first), read_edges(a.neg, a.parent_first)
        print(json.dumps({**m.evaluate(pos, neg), "ranking": m.rank(pos)}))
    elif a.cmd == "info":
        print(json.dumps(m.summary(), indent=1))
    elif a.cmd == "export":
        m.export(a.out)
    elif a.cmd == "plot":
        from .plot import draw
        draw(m, a.out, dims=tuple(a.dims), labels=a.labels)


if __name__ == "__main__":
    main()
