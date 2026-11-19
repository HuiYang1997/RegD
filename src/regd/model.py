"""The user-facing RegD class."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch

from . import geometry as G
from .evaluate import f1_at, mean_rank_metrics
from .graph import Hierarchy, as_edge_list
from .training import encode_texts, score_pairs, train


def _opt(default, help, choices=None):
    return field(default=default, metadata={"help": help, "choices": choices})


@dataclass
class Config:
    """All options of a model. `regd options` prints this table; RegD(**options) sets them on top of a preset."""
    # ---- regions
    dim: int = _opt(5, "dimension of the space (a node has 2 * dim numbers)")
    region: str = _opt("box", "shape of a node's region", G.REGIONS)
    boundary: str = _opt("distance", "boundary distance b: signed distance to the parent's boundary, or (boxes) the "
                         "paper's 'hybrid' log-volume of the part of the child outside the parent", G.BOUNDARIES)
    # ---- energy and decision
    energy: str = _opt("paper", "paper: b + lam*d;  poly: (1+b)(1+lam*d) - 1;  exp: e^b (1+lam*d) - 1", G.ENERGIES)
    depth: str = _opt("ratio", "depth dissimilarity d: the paper's ratio, the metric log(1+sqrt(q)), or plain L1",
                      G.DEPTHS)
    lam: float = _opt(None, "weight of the depth term; default 0.5 up to dim 5 and 0 above")
    p: int = _opt(None, "norm in the paper's d (depth=ratio); default 1 for boxes, 2 for balls", (1, 2))
    rule: str = _opt("transitive", "transitive: accept when score <= tau with tau <= 0 (score = b for the paper "
                     "energy, E for poly / exp);  tuned: the paper's free threshold on b + lam*d",
                     ("transitive", "tuned"))
    # ---- loss
    loss: str = _opt("cross", "cross: the paper's contrastive loss;  bce;  hinge", ("cross", "bce", "hinge"))
    margin: float = _opt(0.01, "negatives are pushed to E >= margin (cross, hinge)")
    pos_margin: float = _opt(0.01, "positives are pushed to E <= -pos_margin")
    temp: float = _opt(0.1, "temperature of the bce loss")
    # ---- negatives
    negatives: str = _opt("fresh", "fresh: re-drawn every epoch;  fixed: drawn once (the paper's way)",
                          ("fresh", "fixed"))
    neg_filter: str = _opt("closure", "pairs never used as negatives: the transitive closure of the training edges, "
                           "only the edges themselves (told), or nothing (none)", ("closure", "told", "none"))
    neg_side: str = _opt("both", "corrupt the child, the parent or both sides of an edge", ("both", "parent", "child"))
    n_neg: int = _opt(9, "negatives per edge and side")
    hard_frac: float = _opt(0.0, "share of hard negatives: wrong nodes with the lowest current energy "
                            "(needs neg_filter closure or told)")
    hard_pool: int = _opt(256, "size of the random candidate pool for hard negatives (all nodes if the graph is smaller)")
    # ---- optimisation
    epochs: int = _opt(400, "passes over the edges")
    batch_size: int = _opt(None, "default: number of edges / 36, between 8 and 2048")
    lr: float = _opt(None, "Adam learning rate; default 0.005 for boxes, 0.01 for balls")
    schedule: str = _opt("const", "learning-rate schedule: constant, or 5% warm-up + cosine decay", ("const", "cosine"))
    eval_every: int = _opt(20, "validate (and keep the best checkpoint) every this many epochs")
    patience: int = _opt(None, "stop after this many validations without improvement; default: never")
    seed: int = _opt(43, "random seed")
    # ---- initialisation
    init: str = _opt("random", "random: the paper's initialisation;  text: projected frozen sentence embeddings of "
                     "the node names", ("random", "text"))
    encoder: str = _opt("BAAI/bge-base-en-v1.5", "sentence-transformers model for init=text")
    device: str = _opt(None, "cpu, cuda, ...; default cuda if available")

    def resolve(self):
        for f in fields(self):
            ch, v = f.metadata["choices"], getattr(self, f.name)
            if ch and v is not None and v not in ch:
                raise ValueError(f"{f.name}={v!r}: must be one of {list(ch)}")
        if self.boundary == "volume" and self.region != "box":
            raise ValueError("boundary='volume' is defined for boxes only")
        if self.hard_frac > 0 and self.neg_filter == "none":
            raise ValueError("hard negatives need neg_filter='closure' or 'told' (otherwise true pairs are mined)")
        if self.lam is None:
            self.lam = 0.5 if self.dim <= 5 else 0.0
        if self.p is None:
            self.p = 1 if self.region == "box" else 2
        if self.lr is None:
            self.lr = 0.005 if self.region == "box" else 0.01
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self


def options_table():
    """The options as text: name, default, choices, meaning."""
    rows = []
    for f in fields(Config):
        ch = f.metadata["choices"]
        rows.append(f"{f.name:<11} {str(f.default):<24} {(' | '.join(map(str, ch)) if ch else ''):<26} "
                    f"{f.metadata['help']}")
    return "\n".join(rows)


PRESETS = {
    # the published protocol: negatives drawn once and unfiltered, margins 0 / 0.001, threshold tuned on validation
    "paper": dict(loss="cross", margin=0.001, pos_margin=0.0, negatives="fixed", neg_filter="none", rule="tuned"),
    # 'T1x': the paper's loss with margins 0.01, negatives re-drawn every epoch and filtered by the training closure
    "strong": dict(loss="cross", margin=0.01, pos_margin=0.01, negatives="fresh", neg_filter="closure",
                   rule="transitive"),
    # 'T2x': the same negatives with a BCE loss (better on tree-like data such as WordNet)
    "strong-bce": dict(loss="bce", temp=0.1, pos_margin=0.0, negatives="fresh", neg_filter="closure",
                       rule="transitive"),
    # for incomplete graphs (missing edges are to be predicted): nothing is assumed about pairs that are not edges
    "predict": dict(loss="bce", temp=0.1, pos_margin=0.0, negatives="fresh", neg_filter="told", rule="transitive"),
}


class RegD:
    """Region embedding of a hierarchy: every node becomes a box or a ball, 'x is-a y' becomes containment.

        model = RegD(dim=5, region="box").fit([("dog", "mammal"), ("mammal", "animal"), ...])
        model.is_a("dog", "animal");  model.ancestors("dog");  model.save("m.pt")
    """

    def __init__(self, dim=5, region="box", preset="strong", **options):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        known = {f.name for f in fields(Config)}
        unknown = set(options) - known
        if unknown:
            raise TypeError(f"unknown options {sorted(unknown)}; available: {sorted(known)}")
        self.config = Config(dim=dim, region=region, **{**PRESETS[preset], **options}).resolve()
        self.preset = preset
        self.hierarchy = None
        self.weight = None
        self.tau = 0.0
        self.history = []

    # ------------------------------------------------------------------ training
    def fit(self, edges, valid=None, nodes=None, parent_first=False, verbose=True, texts=None, init_vectors=None):
        """Train on is-a edges.

        edges   list of (child, parent) pairs, a networkx DiGraph (u -> v = 'u is-a v') or a path to a two-column file
        valid   optional (positive pairs, negative pairs), each a list of (child, parent); used to pick the checkpoint
                and the threshold. Without it, validation pairs are drawn from the transitive closure of the edges.
        nodes   optional extra node names (nodes that occur in no training edge)
        texts   init="text": optional dict node -> text to encode (default: the node name, '_' replaced by ' ')
        init_vectors   optional dict node -> vector (or an [N, D] array in the order of model.hierarchy.nodes when
                the node order is known): precomputed embeddings used instead of running the encoder
        """
        extra = list(nodes or [])
        if valid is not None:
            valid = [as_edge_list(v, parent_first) for v in valid]
            extra += [n for v in valid for pair in v for n in pair]
        self.hierarchy = h = Hierarchy(edges, nodes=extra, parent_first=parent_first)
        if valid is None:
            vp, vn = h.eval_pairs(seed=self.config.seed)
        else:
            vp, vn = (self._pairs(v) for v in valid)
        if init_vectors is not None:
            if isinstance(init_vectors, dict):
                init_vectors = np.stack([np.asarray(init_vectors[n]) for n in h.nodes])
            if len(init_vectors) != h.N:
                raise ValueError(f"init_vectors has {len(init_vectors)} rows for {h.N} nodes")
        elif self.config.init == "text":
            names = [(texts or {}).get(n, str(n).replace("_", " ")) for n in h.nodes]
            init_vectors = encode_texts(names, self.config.encoder, device=self.config.device)
        elif self.config.init != "random":
            raise ValueError("init must be 'random' or 'text'")
        self.weight, self.tau, self.history = train(h, self.config, vp, vn, verbose=verbose,
                                                    init_vectors=init_vectors)
        return self

    # ------------------------------------------------------------------ queries
    def _check(self):
        if self.weight is None:
            raise RuntimeError("call fit() or load() first")

    def _pairs(self, pairs):
        """(child, parent) names -> [n, 2] array of (parent, child) ids."""
        pairs = list(pairs)
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([self.hierarchy.ids([p for _, p in pairs]), self.hierarchy.ids([c for c, _ in pairs])], 1)

    def score(self, child, parent):
        """Decision score of 'child is-a parent' (lower = more likely; accepted when <= model.tau).
        Takes two names or two equally long lists of names."""
        self._check()
        single = not isinstance(child, (list, tuple, np.ndarray))
        pairs = self._pairs([(child, parent)] if single else zip(child, parent))
        s = score_pairs(self.weight, torch.as_tensor(pairs, device=self.weight.device), self.config).cpu().numpy()
        return float(s[0]) if single else s

    def is_a(self, child, parent):
        s = self.score(child, parent)
        return bool(s <= self.tau) if np.isscalar(s) else s <= self.tau

    @torch.no_grad()
    def _score_all(self, ids, as_child=True):
        """[len(ids), N] scores of every node as parent of ids (as_child) or as child of ids."""
        w, x = self.weight, self.weight[torch.as_tensor(ids, device=self.weight.device)][:, None]
        return G.decision_score(w[None], x, self.config) if as_child else G.decision_score(x, w[None], self.config)

    def _neighbours(self, node, k, as_child):
        self._check()
        i = int(self.hierarchy.ids([node])[0])
        s = self._score_all([i], as_child)[0].cpu().numpy()
        s[i] = np.inf
        order = np.argsort(s, kind="stable")
        order = order[:k] if k is not None else order[s[order] <= self.tau]
        return [(self.hierarchy.nodes[j], float(s[j])) for j in order]

    def ancestors(self, node, k=None):
        """Predicted ancestors of node as (name, score), best first: all accepted ones, or the k best if k is given."""
        return self._neighbours(node, k, as_child=True)

    def descendants(self, node, k=None):
        return self._neighbours(node, k, as_child=False)

    def regions(self):
        """dict with the node names, the centres [N, dim] and the box offsets [N, dim] / ball radii [N]."""
        self._check()
        c, o = G.split(self.weight, self.config.dim)
        o = o if self.config.region == "box" else o[:, 0]
        return {"nodes": list(self.hierarchy.nodes), "center": c.cpu().numpy(),
                "offset" if self.config.region == "box" else "radius": o.cpu().numpy()}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, positives, negatives):
        """Precision / recall / F1 of is_a on (child, parent) pairs, with the threshold chosen in fit()."""
        self._check()
        dev = self.weight.device
        sp = score_pairs(self.weight, torch.as_tensor(self._pairs(positives), device=dev), self.config).cpu().numpy()
        sn = score_pairs(self.weight, torch.as_tensor(self._pairs(negatives), device=dev), self.config).cpu().numpy()
        return f1_at(np.r_[sp, sn], np.r_[np.ones(len(sp)), np.zeros(len(sn))], self.tau)

    def rank(self, pairs, chunk=256):
        """Ranking metrics: for every (child, parent) pair, the rank of the parent among all nodes as parents of the
        child. The child itself and its other training-closure ancestors are left out of the ranking."""
        self._check()
        pr = self._pairs(pairs)
        anc = self.hierarchy.ancestors_of()
        ranks = []
        for s in range(0, len(pr), chunk):
            part = pr[s:s + chunk]
            sc = self._score_all(part[:, 1], as_child=True).cpu().numpy()
            for row, (p, c) in zip(sc, part):
                true = row[p]
                row[[p, c]] = np.inf
                known = anc.get(int(c))
                if known is not None:
                    row[known] = np.inf
                ranks.append(1 + (row < true).sum() + 0.5 * (row == true).sum())
        return mean_rank_metrics(ranks)

    def containment(self):
        """Share of the training edges whose child region lies inside the parent region (b <= 0). When this is low
        the regions do not nest and is_a behaves like a nearest-neighbour test."""
        self._check()
        e = torch.as_tensor(self.hierarchy.edges, device=self.weight.device)
        with torch.no_grad():
            b, _ = G.components(self.weight[e[:, 0]], self.weight[e[:, 1]], self.config.dim, self.config.region,
                                need_depth=False)
        return float((b <= 0).float().mean())

    def transitivity_violations(self, max_nodes=2000, seed=0):
        """Share of predicted chains a is-a b, b is-a c for which a is-a c is NOT predicted (on a node sample)."""
        self._check()
        n = self.hierarchy.N
        ids = np.arange(n) if n <= max_nodes else np.random.default_rng(seed).choice(n, max_nodes, replace=False)
        w = self.weight[torch.as_tensor(ids, device=self.weight.device)]
        with torch.no_grad():
            A = torch.cat([G.decision_score(w[None], w[s:s + 256, None], self.config) <= self.tau
                           for s in range(0, len(ids), 256)]).float()  # A[child, parent]
        A.fill_diagonal_(0)
        chains = A @ A
        total = chains.sum().item()
        return float((chains * (1 - A)).sum().item() / total) if total else 0.0

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        self._check()
        torch.save({"format": "regd-1", "config": asdict(self.config), "preset": self.preset, "tau": self.tau,
                    "nodes": list(self.hierarchy.nodes), "edges": self.hierarchy.edges,
                    "weight": self.weight.cpu(), "history": self.history}, path)

    def export(self, path):
        """Write the regions as a table: .npz (arrays nodes / center / offset or radius) or text (.tsv: one node per
        line, name, then the centre, then the offsets or the radius)."""
        r = self.regions()
        size = r.get("offset", r.get("radius"))
        if str(path).endswith(".npz"):
            np.savez(path, nodes=np.array([str(n) for n in r["nodes"]]), center=r["center"],
                     **{"offset" if "offset" in r else "radius": size}, tau=self.tau)
            return
        with open(path, "w", encoding="utf8") as f:
            for name, c, o in zip(r["nodes"], r["center"], size):
                f.write("\t".join([str(name)] + [f"{x:.6g}" for x in np.r_[c, np.atleast_1d(o)]]) + "\n")

    def summary(self):
        """Size of the graph, the options that differ from the defaults of Config, and the validation result."""
        self._check()
        base = Config(dim=self.config.dim, region=self.config.region).resolve()
        changed = {k: v for k, v in asdict(self.config).items() if getattr(base, k) != v and k != "device"}
        return {"nodes": self.hierarchy.N, "edges": int(len(self.hierarchy.edges)), "dim": self.config.dim,
                "region": self.config.region, "preset": self.preset, "options": changed, "tau": self.tau,
                "val_f1": max((h["val_f1"] for h in self.history), default=None),
                "epochs_run": self.history[-1]["epoch"] if self.history else 0, "containment": self.containment()}

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls.__new__(cls)
        m.config = Config(**{**d["config"], "device": device})
        m.preset, m.tau, m.history = d["preset"], d["tau"], d["history"]
        nodes = d["nodes"]
        m.hierarchy = Hierarchy([(nodes[c], nodes[p]) for p, c in d["edges"]], nodes=nodes)
        m.weight = d["weight"].to(device)
        return m
