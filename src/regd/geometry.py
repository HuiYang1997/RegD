"""Region geometry of RegD: boundary distance b, depth dissimilarity d and the energies built from them.

Every node is one row of 2*dim numbers: a centre (first dim) and raw offsets (last dim, passed through softplus).
A box uses all offsets (half side lengths); a ball uses the first offset as its radius (the storage layout of the
original RegD code is kept so that its initialisation carries over).

    b(parent, child) <= 0   <=>   the child region lies inside the parent region.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

REGIONS = ("box", "ball")
BOUNDARIES = ("distance", "volume")
DEPTHS = ("ratio", "logsqrt", "l1")
ENERGIES = ("paper", "poly", "exp")
SHRINK = 1e-4  # in training the parent region is shrunk by exp(-SHRINK), as in the original code


def split(emb: torch.Tensor, dim: int):
    """Row(s) of the embedding table -> (centre, positive offsets)."""
    return emb[..., :dim], F.softplus(emb[..., dim:])


def _log_volume(lo, hi):
    return torch.log(torch.clamp(hi - lo, min=0.0) + 1e-6).sum(dim=-1)


def components(parent, child, dim, region="box", depth="ratio", p=1, training=False, need_depth=True,
               boundary="distance"):
    """Boundary distance b and depth dissimilarity d for broadcastable parent / child rows.

    boundary distance  signed distance of the child region to the parent's boundary (negative inside)
             volume    boxes only, the paper's 'hybrid' option: log-volume of the child minus log-volume of its
                       intersection with the parent (0 when contained, never negative)
    depth    ratio     the paper's d: ||P1 - P2||_p^p / (size_parent * size_child)
             logsqrt   log(1 + sqrt(q)), q = the p = 2 ratio; a metric (proved for balls)
             l1        plain L1 distance between the regions' parameters
    """
    pc, po = split(parent, dim)
    cc, co = split(child, dim)
    if training:
        po = po * math.exp(-SHRINK)
    dc = pc - cc
    if region == "box":
        if boundary == "volume":
            lo = torch.maximum(pc - po, cc - co)
            hi = torch.minimum(pc + po, cc + co)
            b = _log_volume(cc - co, cc + co) - _log_volume(lo, hi)
        elif boundary == "distance":
            diff = dc.abs() - po + co
            outside = torch.norm(F.relu(diff), dim=-1)
            b = torch.where(outside > 0, outside, diff.max(dim=-1)[0])
        else:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if not need_depth:
            return b, None
        do = po - co
        if depth == "l1":
            d = (dc.abs() + do.abs()).mean(-1)
        else:
            size = po.norm(dim=-1) * co.norm(dim=-1)
            if depth == "ratio" and p == 1:
                d = (dc.abs().sum(-1) + do.abs().sum(-1)) / size
            else:
                d = (dc.square().sum(-1) + do.square().sum(-1)) / size
    elif region == "ball":
        if boundary != "distance":
            raise ValueError("boundary='volume' is defined for boxes only")
        rp, rc = po[..., 0], co[..., 0]
        b = torch.norm(dc, dim=-1) - rp + rc
        if not need_depth:
            return b, None
        dr = rp - rc
        if depth == "l1":
            d = torch.sqrt(dc.square().sum(-1) + 1e-12) + dr.abs()
        elif depth == "ratio" and p == 1:
            d = (dc.abs().sum(-1) + dr.abs()) / (rp * rc)
        else:
            d = (dc.square().sum(-1) + dr.square()) / (rp * rc)
    else:
        raise ValueError(f"region must be one of {REGIONS}")
    if depth == "logsqrt":
        d = torch.log1p(torch.sqrt(d + 1e-12))
    elif depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    return b, d


def combine(b, d, lam, energy="paper", training=False):
    """Energy from b and d.

    paper  b + lam * d   (in training the depth term is switched off for pairs that are already contained)
    poly   (1 + b)(1 + lam * d) - 1
    exp    e^b (1 + lam * d) - 1
    """
    if d is None or lam == 0:
        if energy == "exp":
            return torch.expm1(torch.clamp(b, max=20.0))
        return b
    if energy == "paper":
        if training:
            return b + lam * torch.where(b < 0, torch.zeros_like(b), d)
        return b + lam * d
    if energy == "poly":
        return (1.0 + b) * (1.0 + lam * d) - 1.0
    if energy == "exp":
        return torch.exp(torch.clamp(b, max=20.0)) * (1.0 + lam * d) - 1.0
    raise ValueError(f"energy must be one of {ENERGIES}")


def training_energy(parent, child, cfg):
    b, d = components(parent, child, cfg.dim, cfg.region, cfg.depth, cfg.p, training=True,
                      need_depth=cfg.lam != 0, boundary=cfg.boundary)
    return combine(b, d, cfg.lam, cfg.energy, training=True)


def decision_score(parent, child, cfg):
    """The quantity compared with the threshold tau (lower = more likely child is-a parent).

    rule transitive  paper energy: b alone (the depth term only shapes training);  poly / exp: the energy itself.
                     With tau <= 0 the accepted relation is transitive.
    rule tuned       the paper's protocol: evaluation energy b + lam * d with a free threshold (not transitive).
    """
    gate = cfg.rule == "transitive" and cfg.energy == "paper"
    b, d = components(parent, child, cfg.dim, cfg.region, cfg.depth, cfg.p, training=False,
                      need_depth=cfg.lam != 0 and not gate, boundary=cfg.boundary)
    if gate:
        return b
    return combine(b, d, cfg.lam, cfg.energy, training=False)
