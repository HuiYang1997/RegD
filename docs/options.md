# All options

Most users only need the few options in the [main README](../README.md#choosing-the-settings). This page lists
every option. `regd options` (or `regd.options_table()`) prints the same list.

Set an option as a keyword, `RegD(dim=10, preset="strong-bce", lam=0, epochs=200)`, or on the command line with
`--set lam=0 epochs=200`. Options given explicitly override the preset.

## Presets

A preset is a named set of option values.

| preset | loss | negatives | decision rule |
|---|---|---|---|
| `strong` (default) | the paper's loss, margins 0.01 | re-drawn every epoch; never a pair from the transitive closure of the training edges | `b <= tau`, `tau <= 0` (transitive) |
| `strong-bce` | BCE, temperature 0.1 | as above | as above |
| `predict` | BCE | re-drawn every epoch; only the training edges themselves are excluded | as above |

## Regions

| option | default | choices | meaning |
|---|---|---|---|
| `dim` | 5 | | dimension of the space; a node has `2 * dim` numbers |
| `region` | `box` | `box`, `ball` | shape of a node's region |
| `boundary` | `distance` | `distance`, `volume` | `b` as the signed distance to the parent's boundary, or (boxes) the paper's "hybrid" log-volume of the part of the child outside the parent |

## Energy and decision

| option | default | choices | meaning |
|---|---|---|---|
| `energy` | `paper` | `paper`, `poly`, `exp` | `b + lam d`; `(1+b)(1+lam d) - 1`; `e^b (1+lam d) - 1` |
| `depth` | `ratio` | `ratio`, `logsqrt`, `l1` | depth dissimilarity `d`: the paper's ratio, the metric `log(1 + sqrt(q))`, plain L1 |
| `lam_set` | `[0.5]` | | weights of the depth term to try; with several values `fit` trains once per value and keeps the one with the best validation F1 (`model.lam_results` lists them) |
| `lam` | | | short form: `lam=x` is `lam_set=[x]` (one training run). After `fit`, `model.config.lam` is the value in use |
| `p` | 1 (box), 2 (ball) | 1, 2 | norm in the paper's `d` |
| `rule` | `transitive` | `transitive`, `tuned` | accept when score `<= tau` with `tau <= 0` (score = `b` for the paper energy, `E` for `poly` / `exp`); or the paper's free threshold on `b + lam d` |

## Loss

| option | default | choices | meaning |
|---|---|---|---|
| `loss` | `cross` | `cross`, `bce`, `hinge` | the paper's contrastive loss; BCE; hinge |
| `margin` | 0.01 | | negatives are pushed to `E >= margin` (`cross`, `hinge`) |
| `pos_margin` | 0.01 | | positives are pushed to `E <= -pos_margin` |
| `temp` | 0.1 | | temperature of `bce` |

## Negatives

| option | default | choices | meaning |
|---|---|---|---|
| `negatives` | `fresh` | `fresh`, `fixed` | re-drawn every epoch, or drawn once |
| `neg_filter` | `closure` | `closure`, `told`, `none` | pairs never used as negatives: the transitive closure of the training edges, only the edges, or nothing |
| `neg_side` | `both` | `both`, `parent`, `child` | which side of an edge is corrupted |
| `n_neg` | 9 | | negatives per edge and side |
| `hard_frac` | 0 | | share of hard negatives (wrong nodes with the lowest current energy) |
| `hard_pool` | 256 | | random candidate pool for hard negatives |

## Optimisation and initialisation

| option | default | choices | meaning |
|---|---|---|---|
| `epochs` | 400 | | passes over the edges |
| `batch_size` | edges / 36, in [8, 2048] | | |
| `lr` | 0.005 (box), 0.01 (ball) | | Adam learning rate |
| `schedule` | `const` | `const`, `cosine` | constant, or 5% warm-up + cosine decay |
| `eval_every` | 20 | | validate and keep the best checkpoint every so many epochs |
| `patience` | none | | stop after this many validations without improvement |
| `seed` | 43 | | |
| `init` | `random` | `random`, `text` | the paper's initialisation, or projected frozen sentence embeddings of the node names |
| `encoder` | `BAAI/bge-base-en-v1.5` | | sentence-transformers model for `init="text"` |
| `device` | cuda if available | | |

## Notes

**Which negative filter?** `closure` is right when the graph is the hierarchy you want to embed: every entailed pair is known,
so none of them should be pushed apart. If you train on an incomplete graph and want to predict missing edges, use
`preset="predict"` (`neg_filter="told"`); the closure of an incomplete graph says nothing about the missing part.

**Text initialisation.** Needs `pip install ".[text]"` (sentence-transformers). `RegD(init="text")` encodes the node names once with a frozen sentence-transformers model,
projects the vectors to the table size with a fixed random matrix and normalises the rows. Training is unchanged and
the table is free afterwards. `fit(..., texts={node: "text"})` supplies other texts than the names;
`fit(..., init_vectors={node: vector})` supplies precomputed vectors and needs no encoder.

**The transitive rule and `boundary="volume"`.** The volume boundary is never negative, so with the transitive rule
`tau` can only be 0. Use it with `rule="tuned"`.
