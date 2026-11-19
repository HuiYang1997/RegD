"""A hierarchy given as is-a edges over arbitrary hashable node names."""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np


def read_edges(path, parent_first=False, delimiter=None):
    """Read 'child<sep>parent' lines (tab or whitespace separated; '#' lines and blank lines skipped)."""
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter else (line.split("\t") if "\t" in line else line.split())
            a, b = parts[0].strip(), parts[1].strip()
            out.append((b, a) if parent_first else (a, b))
    return out


def as_edge_list(edges, parent_first=False):
    """Accept a list of (child, parent) pairs, a networkx DiGraph (edge u -> v means 'u is-a v') or a file path."""
    if isinstance(edges, (str, bytes)) or hasattr(edges, "__fspath__"):
        return read_edges(edges, parent_first=parent_first)
    if hasattr(edges, "edges") and hasattr(edges, "nodes"):  # networkx
        edges = list(edges.edges())
    edges = [(e[0], e[1]) for e in edges]
    return [(b, a) for a, b in edges] if parent_first else edges


class Hierarchy:
    """Node table, training edges and their transitive closure.

    Internally an edge is the id pair (parent, child) and a pair is encoded as the key parent * N + child.
    """

    def __init__(self, edges, nodes=None, parent_first=False):
        edges = as_edge_list(edges, parent_first)
        self.nodes = []
        self.index = {}
        for n in (nodes or []):
            self._add(n)
        for c, p in edges:
            self._add(c)
            self._add(p)
        self.N = len(self.nodes)
        pairs = {(self.index[p], self.index[c]) for c, p in edges if p != c}
        if not pairs:
            raise ValueError("no edges")
        self.edges = np.array(sorted(pairs), dtype=np.int64)  # [E, 2] = (parent, child)
        self._closure = None

    def _add(self, n):
        if n not in self.index:
            self.index[n] = len(self.nodes)
            self.nodes.append(n)

    def ids(self, names):
        try:
            return np.array([self.index[n] for n in names], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"unknown node {e.args[0]!r}") from None

    def keys(self, pairs):
        """Keys of (parent, child) id pairs."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        return pairs[:, 0] * self.N + pairs[:, 1]

    @property
    def edge_keys(self):
        return np.unique(self.keys(self.edges))

    @property
    def closure_keys(self):
        """Sorted keys of every (ancestor, descendant) pair entailed by the edges."""
        if self._closure is None:
            children = defaultdict(list)
            for p, c in self.edges:
                children[int(p)].append(int(c))
            chunks, cyclic = [], 0
            for u in children:
                seen, stack = set(), [u]
                while stack:
                    x = stack.pop()
                    for y in children.get(x, ()):
                        if y not in seen:
                            seen.add(y)
                            stack.append(y)
                if u in seen:
                    cyclic += 1
                    seen.discard(u)
                chunks.append(u * self.N + np.fromiter(seen, dtype=np.int64, count=len(seen)))
            if cyclic:
                warnings.warn(f"the graph has cycles ({cyclic} nodes are their own ancestor); nodes on a cycle are "
                              "treated as mutual ancestors", stacklevel=2)
            self._closure = np.unique(np.concatenate(chunks))
        return self._closure

    def ancestors_of(self):
        """dict child id -> sorted array of ancestor ids (from the closure)."""
        k = self.closure_keys
        par, ch = k // self.N, k % self.N
        order = np.argsort(ch, kind="stable")
        par, ch = par[order], ch[order]
        cut = np.flatnonzero(np.diff(ch)) + 1
        return {int(c[0]): p for c, p in zip(np.split(ch, cut), np.split(par, cut))}

    def eval_pairs(self, n_pos=None, neg_ratio=10, seed=0):
        """Validation pairs made from the graph itself (used when the caller supplies none).

        Positives: entailed pairs that are not direct edges (direct edges if there are too few of those).
        Negatives: positives with the child or the parent replaced by a random node, outside the closure.
        Returns (pos, neg) as [n, 2] arrays of (parent, child) ids.
        """
        rng = np.random.default_rng(seed)
        clo = self.closure_keys
        cand = np.setdiff1d(clo, self.edge_keys, assume_unique=True)
        if len(cand) < 20:
            cand = clo
        if n_pos is None:
            n_pos = int(min(len(cand), max(200, min(10000, 0.1 * len(cand)))))
        pos_k = rng.choice(cand, size=min(n_pos, len(cand)), replace=False)
        pos = np.stack([pos_k // self.N, pos_k % self.N], 1)
        neg = np.repeat(pos, neg_ratio, axis=0)
        side = rng.integers(0, 2, len(neg))
        todo = np.arange(len(neg))
        keep = np.ones(len(neg), dtype=bool)
        for _ in range(50):
            r = rng.integers(0, self.N, len(todo))
            neg[todo, side[todo]] = r
            k = neg[todo, 0] * self.N + neg[todo, 1]
            bad = (neg[todo, 0] == neg[todo, 1]) | np.isin(k, clo)
            todo = todo[bad]
            if not len(todo):
                break
        keep[todo] = False
        return pos, neg[keep]
