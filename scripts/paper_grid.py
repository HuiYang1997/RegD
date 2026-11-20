"""Commands for the paper's 16 configurations x presets (paper hyper-parameters from scripts/paper_configs.json).
python scripts/paper_grid.py <data dir> [--presets strong strong-bce] > jobs.txt ;  then e.g.
xargs -P 8 -I{} sh -c "{}" < jobs.txt"""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--presets", nargs="*", default=["strong", "strong-bce"])
ap.add_argument("--out", default="runs/grid.jsonl")
ap.add_argument("--only", nargs="*", default=None, help="data sets to include")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
for c in json.load(open(os.path.join(here, "paper_configs.json"))):
    ds = c["dataset"]
    if a.only and ds not in a.only:
        continue
    prefix = f"{a.data}/{'maxn' if ds in ('mammal', 'noun') else ds}/{ds}_closure.tsv"
    seeds = "43 0 1" if ds == "mammal" else "43 0"
    for preset in a.presets:
        opts = [f"lam={c['lam0']}", f"p={c['p']}", f"lr={c['lr']}", f"batch_size={c['batch']}", f"epochs={c['epochs']}"]
        print(f"python {here}/bench.py {prefix} --dim {c['dim']} --region {c['geom']} --preset {preset} "
              f"--seeds {seeds} --tag {c['cfg']} --set {' '.join(opts)} --out {a.out} "
              f"> runs/grid_{c['cfg']}_{preset}.log 2>&1")
