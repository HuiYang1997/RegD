"""Initialisation, negative sampling, losses and the training loop."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from .evaluate import best_threshold


# ---------------------------------------------------------------------------------------------- initial table
def init_table(n, dim, device):
    """The original RegD initialisation: uniform(-1, 1) rows, normalised to unit length."""
    w = torch.empty(n, 2 * dim, device=device).uniform_(-1, 1)
    return torch.nn.Parameter(w / w.norm(dim=1, keepdim=True))


def text_table(vectors, dim, seed, device):
    """Initial table from text embeddings [N, D]: a fixed random projection to 2 * dim, rows normalised to unit
    length (the same scale as the random initialisation). The table is free afterwards; the encoder is not trained."""
    z = torch.as_tensor(np.asarray(vectors), dtype=torch.float32)
    proj = torch.randn(z.shape[1], 2 * dim, generator=torch.Generator().manual_seed(seed)) / z.shape[1] ** 0.5
    w = z @ proj
    return torch.nn.Parameter((w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)).to(device))


def encode_texts(texts, encoder, device=None, batch_size=256):
    """Sentence embeddings of the node texts with a frozen sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("init='text' needs sentence-transformers: pip install 'regd[text]' "
                          "(or pass your own vectors with init_vectors=...)") from e
    model = SentenceTransformer(encoder, device=device)
    return model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)


# ---------------------------------------------------------------------------------------------- negatives
def member(keys, known):
    """keys in known, for a sorted 1-D tensor known (faster than torch.isin on large closures)."""
    pos = torch.searchsorted(known, keys.reshape(-1)).clamp_(max=known.numel() - 1)
    return (known[pos] == keys.reshape(-1)).view(keys.shape)


@torch.no_grad()
def sample_negatives(weight, rel, n, cfg, known=None):
    """Corrupted children of each parent and corrupted parents of each child.

    rel: [E, 2] (parent, child).  Returns negc, negp ([E, K] node ids) and boolean masks of the usable entries.
    known: sorted keys parent * N + child that must not be used as negatives (None = no filtering, the paper's way).
    cfg.hard_frac > 0: that share of the K negatives are the wrong nodes with the lowest current training energy in
    a random pool of cfg.hard_pool nodes.
    """
    dev, E, K = rel.device, rel.size(0), cfg.n_neg
    u, v = rel[:, :1], rel[:, 1:]
    negc = torch.randint(0, n, (E, K), device=dev)
    negp = torch.randint(0, n, (E, K), device=dev)
    okc = torch.ones(E, K, dtype=torch.bool, device=dev)
    okp = torch.ones(E, K, dtype=torch.bool, device=dev)
    if known is None:
        return negc, negp, okc, okp

    def bad(neg, side):
        keys = u * n + neg if side == "child" else neg * n + v
        return (neg == u) | (neg == v) | member(keys, known)

    for _ in range(20):
        bc, bp = bad(negc, "child"), bad(negp, "parent")
        if not (bc.any() or bp.any()):
            break
        negc = torch.where(bc, torch.randint(0, n, negc.shape, device=dev), negc)
        negp = torch.where(bp, torch.randint(0, n, negp.shape, device=dev), negp)
    okc, okp = ~bad(negc, "child"), ~bad(negp, "parent")

    k_hard = int(round(K * cfg.hard_frac))
    if k_hard > 0:
        for s in range(0, E, 256):
            uu, vv = rel[s:s + 256, 0], rel[s:s + 256, 1]
            pool = torch.arange(n, device=dev) if n <= cfg.hard_pool else torch.randint(0, n, (cfg.hard_pool,), device=dev)
            k = min(k_hard, pool.numel())
            wp = weight[pool][None]
            for side in ("child", "parent"):
                if side == "child":
                    e = G.training_energy(weight[uu][:, None], wp, cfg)
                    keys = uu[:, None] * n + pool[None]
                else:
                    e = G.training_energy(wp, weight[vv][:, None], cfg)
                    keys = pool[None] * n + vv[:, None]
                mask = (pool[None] == uu[:, None]) | (pool[None] == vv[:, None]) | member(keys, known)
                e = torch.nan_to_num(e, nan=float("inf")).masked_fill(mask, float("inf"))
                top = torch.topk(e, k, dim=1, largest=False)
                found = ~torch.isinf(top.values)
                tgt, ok = (negc, okc) if side == "child" else (negp, okp)
                tgt[s:s + 256, :k] = torch.where(found, pool[top.indices], tgt[s:s + 256, :k])
                ok[s:s + 256, :k] |= found
    return negc, negp, okc, okp


# ---------------------------------------------------------------------------------------------- losses
def loss_fn(pos, neg, ok, cfg):
    """pos: [B] energies of true pairs; neg: [B, K] energies of corrupted pairs; ok: usable negatives.

    cross  the paper's contrastive loss: -log( e^-[E+ + m+]_+ / (e^-[E+ + m+]_+ + sum_k e^[m- - E-_k]_+) )
    bce    softplus((E+ + m+) / T) + mean_k softplus(-E-_k / T)
    hinge  [E+ + m+]_+ + mean_k [m- - E-_k]_+
    """
    okf = ok.to(pos.dtype)
    count = okf.sum(-1).clamp(min=1.0)
    if cfg.loss == "cross":
        p = torch.clamp(pos + cfg.pos_margin, min=0.0)
        q = torch.clamp(neg - cfg.margin, max=0.0)
        return (p + torch.log(torch.exp(-p) + (okf * torch.exp(-q)).sum(-1))).mean()
    if cfg.loss == "bce":
        return (F.softplus((pos + cfg.pos_margin) / cfg.temp) + (okf * F.softplus(-neg / cfg.temp)).sum(-1) / count).mean()
    if cfg.loss == "hinge":
        return (F.relu(pos + cfg.pos_margin) + (okf * F.relu(cfg.margin - neg)).sum(-1) / count).mean()
    raise ValueError("loss must be 'cross', 'bce' or 'hinge'")


@torch.no_grad()
def score_pairs(weight, pairs, cfg, chunk=65536):
    """Decision scores of (parent, child) id pairs."""
    out = []
    for s in range(0, len(pairs), chunk):
        pr = pairs[s:s + chunk]
        out.append(G.decision_score(weight[pr[:, 0]], weight[pr[:, 1]], cfg))
    return torch.cat(out) if out else torch.empty(0, device=weight.device)


# ---------------------------------------------------------------------------------------------- training loop
def train(hier, cfg, val_pos, val_neg, verbose=True, init_vectors=None):
    """Train a free region table on hier.edges.  Returns (weight, tau, history).
    init_vectors: optional [N, D] text embeddings of the nodes for the initial table (see text_table)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    n = hier.N
    weight = init_table(n, cfg.dim, dev) if init_vectors is None else text_table(init_vectors, cfg.dim, cfg.seed, dev)
    opt = torch.optim.Adam([weight], lr=cfg.lr)
    rel = torch.as_tensor(hier.edges, device=dev)
    E = rel.size(0)
    bs = cfg.batch_size or int(min(2048, max(8, E // 36)))
    steps_total = cfg.epochs * math.ceil(E / bs)
    sched = None
    if cfg.schedule == "cosine":  # 5% linear warm-up, then cosine decay to 0
        warm = max(1, int(0.05 * steps_total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
    known = None
    if cfg.neg_filter == "closure":
        known = torch.as_tensor(hier.closure_keys, device=dev)
    elif cfg.neg_filter == "told":
        known = torch.as_tensor(hier.edge_keys, device=dev)
    vp = torch.as_tensor(val_pos, device=dev)
    vn = torch.as_tensor(val_neg, device=dev)
    labels = np.r_[np.ones(len(vp)), np.zeros(len(vn))]
    tau_max = 0.0 if cfg.rule == "transitive" else None
    use_c, use_p = cfg.neg_side in ("both", "child"), cfg.neg_side in ("both", "parent")

    best = {"f1": -1.0, "tau": 0.0, "epoch": 0, "weight": weight.detach().clone()}
    history, t0, negs, stale = [], time.time(), None, 0
    for epoch in range(1, cfg.epochs + 1):
        if negs is None or cfg.negatives == "fresh":
            negs = sample_negatives(weight.detach(), rel, n, cfg, known)
        negc, negp, okc, okp = negs
        total = 0.0
        for idx in torch.randperm(E, device=dev).split(bs):
            u, v = weight[rel[idx, 0]], weight[rel[idx, 1]]
            pos = G.training_energy(u, v, cfg)
            loss = 0.0
            if use_c:  # wrong children of the parent
                loss = loss + loss_fn(pos, G.training_energy(u[:, None], weight[negc[idx]], cfg), okc[idx], cfg)
            if use_p:  # wrong parents of the child
                loss = loss + loss_fn(pos, G.training_energy(weight[negp[idx]], v[:, None], cfg), okp[idx], cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(idx)
        if not np.isfinite(total):
            raise FloatingPointError("the loss became non-finite; lower lr or lam")
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            scores = torch.cat([score_pairs(weight.detach(), vp, cfg), score_pairs(weight.detach(), vn, cfg)])
            tau, f1 = best_threshold(scores.cpu().numpy(), labels, tau_max)
            history.append({"epoch": epoch, "loss": total / E, "val_f1": f1, "tau": tau})
            stale += 1
            if f1 > best["f1"]:
                best = {"f1": f1, "tau": tau, "epoch": epoch, "weight": weight.detach().clone()}
                stale = 0
            if verbose:
                print(f"epoch {epoch:4d}  loss {total / E:.4f}  val F1 {f1:.3f}  (best {best['f1']:.3f} "
                      f"@ {best['epoch']})  {time.time() - t0:.0f}s", flush=True)
            if cfg.patience and stale >= cfg.patience:
                if verbose:
                    print(f"stopped: no better validation F1 in {cfg.patience} evaluations", flush=True)
                break
    return best["weight"], float(best["tau"]), history
