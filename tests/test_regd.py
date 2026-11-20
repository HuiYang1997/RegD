import numpy as np
import pytest
import torch

from regd import Hierarchy, RegD
from regd import geometry as G
from regd.evaluate import best_threshold


def tree(depth=4):
    """Complete binary tree; node i has parent (i - 1) // 2."""
    return [(f"n{i}", f"n{(i - 1) // 2}") for i in range(1, 2 ** (depth + 1) - 1)]


def closure_pairs(edges):
    par = dict(edges)
    out = []
    for c in par:
        p = par[c]
        while True:
            out.append((c, p))
            if p not in par:
                break
            p = par[p]
    return out


def test_closure_and_cycle():
    h = Hierarchy([("a", "b"), ("b", "c"), ("c", "d")])
    assert len(h.closure_keys) == 6
    assert sorted(h.ancestors_of()[h.index["a"]]) == sorted(h.ids(["b", "c", "d"]))
    with pytest.warns(UserWarning):
        assert len(Hierarchy([("a", "b"), ("b", "a")]).closure_keys) == 2


def test_box_geometry():
    # parent box [-2, 2]^2, child box [0, 1]^2: inside, 1 away from the nearest face
    inv = lambda x: float(np.log(np.expm1(x)))  # softplus inverse
    parent = torch.tensor([0.0, 0.0, inv(2.0), inv(2.0)])
    child = torch.tensor([0.5, 0.5, inv(0.5), inv(0.5)])
    b, _ = G.components(parent, child, 2, "box", need_depth=False)
    assert b.item() == pytest.approx(-1.0, abs=1e-5)
    b, _ = G.components(child, parent, 2, "box", need_depth=False)   # the other way round: sticks out
    assert b.item() > 0
    far = torch.tensor([5.0, 0.5, inv(0.5), inv(0.5)])               # [4.5, 5.5] x [0, 1]: 3.5 outside in x
    b, _ = G.components(parent, far, 2, "box", need_depth=False)
    assert b.item() == pytest.approx(3.5, abs=1e-5)


def test_best_threshold():
    tau, f1 = best_threshold([-2, -1, 1, 2], [1, 1, 0, 0])
    assert f1 == 1.0 and -1 < tau < 1
    tau, f1 = best_threshold([-2, 1, 0.5, 2], [1, 1, 0, 0], tau_max=0.0)
    assert tau <= 0 and f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize("region", ["box", "ball"])
def test_fit_tree(region, tmp_path):
    edges = tree()
    m = RegD(dim=3, region=region, epochs=200, device="cpu").fit(edges, verbose=False)
    pos = closure_pairs(edges)
    neg = [(p, c) for c, p in pos]                       # reversed pairs are never true in a tree
    res = m.evaluate(pos, neg)
    assert res["f1"] > 0.85
    assert m.containment() > 0.9
    assert m.transitivity_violations() == 0.0            # the transitive rule (tau <= 0)
    assert m.tau <= 0
    assert m.rank(pos)["mrr"] > 0.5
    assert m.is_a("n30", "n0") and not m.is_a("n0", "n30")
    m.save(tmp_path / "m.pt")
    m2 = RegD.load(tmp_path / "m.pt")
    assert m2.score("n30", "n0") == pytest.approx(m.score("n30", "n0"), abs=1e-6)
    assert [n for n, _ in m2.ancestors("n30")] == [n for n, _ in m.ancestors("n30")]


def test_presets_and_options():
    edges = tree(3)
    for preset in ("paper", "strong-bce"):
        RegD(dim=2, preset=preset, epochs=20, device="cpu").fit(edges, verbose=False)
    for energy in ("poly", "exp"):
        m = RegD(dim=2, energy=energy, depth="logsqrt", lam=0.5, epochs=20, device="cpu").fit(edges, verbose=False)
        assert np.isfinite(m.score("n1", "n0"))
    RegD(dim=2, hard_frac=0.5, neg_filter="told", epochs=20, device="cpu").fit(edges, verbose=False)
    with pytest.raises(TypeError):
        RegD(dim=2, nonsense=1)


def test_init_vectors_and_networkx():
    nx = pytest.importorskip("networkx")
    g = nx.DiGraph(tree(3))
    vec = {n: np.random.default_rng(i).normal(size=16) for i, n in enumerate(g.nodes)}
    m = RegD(dim=2, epochs=20, device="cpu").fit(g, init_vectors=vec, verbose=False)
    assert m.hierarchy.N == g.number_of_nodes()


def test_rank_of_node_without_known_ancestors():
    edges = tree(3)
    m = RegD(dim=2, epochs=5, device="cpu").fit(edges[1:], nodes=["n1"], verbose=False)   # n1 -> n0 is hidden
    r = m.rank([("n1", "n0")])
    s = m._score_all(m.hierarchy.ids(["n1"]))[0].numpy()
    i0, i1 = m.hierarchy.index["n0"], m.hierarchy.index["n1"]
    expected = 1 + sum(s[j] < s[i0] for j in range(len(s)) if j not in (i0, i1))
    assert r["mean_rank"] == pytest.approx(expected)


def test_more_options(tmp_path):
    edges = tree(3)
    for kw in (dict(loss="hinge", margin=0.1, pos_margin=0.1), dict(neg_side="parent"), dict(schedule="cosine"),
               dict(boundary="volume", rule="tuned"), dict(preset="predict"), dict(patience=1, eval_every=5)):
        m = RegD(dim=2, device="cpu", **{"epochs": 40, **kw}).fit(edges, verbose=False)
        assert np.isfinite(m.score("n1", "n0"))
    assert m.history[-1]["epoch"] < 40                                   # patience stopped the last run early
    for bad in (dict(region="ball", boundary="volume"), dict(loss="nope"), dict(hard_frac=0.5, neg_filter="none")):
        with pytest.raises(ValueError):
            RegD(dim=2, **bad)
    m.export(tmp_path / "r.tsv")
    m.export(tmp_path / "r.npz")
    assert len(open(tmp_path / "r.tsv").read().splitlines()) == m.hierarchy.N
    assert np.load(tmp_path / "r.npz")["center"].shape == (m.hierarchy.N, 2)
    s = m.summary()
    assert s["nodes"] == 15 and s["options"]["patience"] == 1


def test_cli(tmp_path, capsys):
    from regd.cli import main
    f = tmp_path / "e.tsv"
    f.write_text("".join(f"{c}\t{p}\n" for c, p in tree(3)))
    main(["train", str(f), "--out", str(tmp_path / "m.pt"), "--dim", "2", "--quiet", "--set", "epochs=40", "lam=0"])
    main(["query", str(tmp_path / "m.pt"), "n14", "-k", "3"])
    main(["info", str(tmp_path / "m.pt")])
    main(["options"])
    main(["export", str(tmp_path / "m.pt"), str(tmp_path / "m.tsv")])
    out = capsys.readouterr().out
    assert '"edges": 14' in out and "neg_filter" in out
    pytest.importorskip("matplotlib")
    main(["plot", str(tmp_path / "m.pt"), str(tmp_path / "m.png")])
    assert (tmp_path / "m.png").stat().st_size > 1000


def test_same_numbers_as_original_code():
    """b, d and the energies agree with the original BoxDistBoundary.depthBox (in paper/, or in the folder named by
    REGD_UPSTREAM)."""
    import os, sys
    up = os.environ.get("REGD_UPSTREAM") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
    if not os.path.exists(os.path.join(up, "BoxDistBoundary.py")):
        pytest.skip("original BoxDistBoundary.py not found")
    sys.path.insert(0, up)
    from BoxDistBoundary import depthBox
    from regd.model import Config
    torch.manual_seed(0)
    for region, p, hybrid in (("box", 1, 0), ("box", 2, 0), ("ball", 2, 0), ("ball", 1, 0), ("box", 1, 1)):
        ref = depthBox(num_entities=50, dim=4, margin=0.001, depth_type="norm", device="cpu", mode=region,
                       add_denominator="non", hybrid=hybrid, lamda=0.7, p=p)
        w = ref.emb.weight.detach()
        par, ch = w[torch.randint(0, 50, (200,))], w[torch.randint(0, 50, (200,))]
        cfg = Config(dim=4, region=region, p=p, lam=0.7, rule="tuned",
                     boundary="volume" if hybrid else "distance").resolve()
        ref.eval()
        assert torch.allclose(ref.energy(par, ch), G.decision_score(par, ch, cfg), atol=1e-5)
        ref.train()
        assert torch.allclose(ref.energy(par, ch), G.training_energy(par, ch, cfg), atol=1e-5)
        e = torch.randn(8, 10)                                           # the loss, on arbitrary energies
        cfg.margin, cfg.pos_margin = 0.001, 0.0
        from regd.training import loss_fn
        assert torch.allclose(ref.loss_cross(e), loss_fn(e[:, 0], e[:, 1:], torch.ones(8, 9, dtype=torch.bool), cfg),
                              atol=1e-6)


def test_lam_set(tmp_path):
    edges = tree()
    assert RegD(device="cpu").config.lam_set == (0.5,) and RegD(device="cpu").config.lam == 0.5   # default: one run
    m = RegD(dim=2, lam_set=[0, 0.5], epochs=40, device="cpu")
    assert m.config.lam_set == (0.0, 0.5) and m.config.lam is None
    m.fit(edges, verbose=False)
    assert [r["lam"] for r in m.lam_results] == [0.0, 0.5]
    best = max(m.lam_results, key=lambda r: r["val_f1"])
    assert m.config.lam in (0.0, 0.5) and m.summary()["val_f1"] == best["val_f1"]
    m.save(tmp_path / "m.pt")
    m2 = RegD.load(tmp_path / "m.pt")
    assert m2.config.lam == m.config.lam and m2.lam_results == m.lam_results
    assert m2.score("n5", "n0") == pytest.approx(m.score("n5", "n0"))
    one = RegD(dim=2, lam=0.3, epochs=20, device="cpu")                 # lam=x is short for lam_set=[x]
    assert one.config.lam_set == (0.3,) and one.config.lam == 0.3
    assert len(one.fit(edges, verbose=False).lam_results) == 1
    assert RegD(dim=2, lam_set=[1.0], device="cpu").config.lam == 1.0
    with pytest.raises(TypeError):
        RegD(lam=0.5, lam_set=[0, 0.5])
