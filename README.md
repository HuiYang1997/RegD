# RegD: Hierarchical Embeddings via Dissimilarity between Arbitrary Euclidean Regions

RegD embeds hierarchical data as regions in Euclidean space. Every node of an is-a graph becomes a box or a ball, and
"x is-a y" becomes "the region of x lies inside the region of y".

This repository contains

- **`regd`**, an installable library that trains on **any** graph of is-a edges (described on this page). It keeps
  the model of the paper and adds a stronger training recipe, a transitive decision rule and a few optional
  extensions. What is the same as the paper and what is not is listed [below](#relation-to-the-paper).
- **the original code of the paper** (`train.py`, `BoxDistBoundary.py`, `run.sh`, `EXP_ontology/`), unchanged; see
  [Paper code](#paper-code).

![Embedding visualization](mammal_dim=2_ball.png)

## Install

```bash
pip install git+https://github.com/HuiYang1997/RegD     # needs torch and numpy
pip install .                                           # or from a clone
pip install ".[text]"                                   # also installs sentence-transformers, for init="text"
```

## Use

```python
from regd import RegD

edges = [("dog", "mammal"), ("cat", "mammal"), ("mammal", "animal"), ("bird", "animal")]   # (child, parent)
model = RegD(dim=5, region="box").fit(edges)

model.is_a("dog", "animal")        # True / False
model.score("dog", "animal")       # lower = more likely; accepted when <= model.tau
model.ancestors("dog")             # [(name, score), ...] best first
model.descendants("animal", k=10)
model.regions()                    # node names, centres, offsets (numpy);  model.export("regions.tsv")
model.summary()                    # size, options, validation F1, containment
model.save("animals.pt");  model = RegD.load("animals.pt")
```

`fit` accepts a list of `(child, parent)` pairs, a `networkx.DiGraph` (edge `u -> v` means "u is-a v") or the path
of a file with one `child<TAB>parent` pair per line. Node names can be any hashable values.

```bash
regd train edges.tsv --dim 5 --region box --out model.pt     # --set key=value ... for any other option
regd query model.pt dog              # predicted ancestors (--descendants, -k 10)
regd query model.pt dog animal       # score and decision of one pair
regd eval  model.pt test_pos.tsv test_neg.tsv
regd info  model.pt                  # size, options, validation F1, containment
regd export model.pt regions.tsv     # or .npz
regd plot  model.pt regions.png      # two of the dimensions (needs matplotlib)
regd options                         # every option with default, choices and meaning
```

### Validation data

Training picks the checkpoint and the threshold `tau` on validation pairs. Pass your own with
`fit(edges, valid=(positive_pairs, negative_pairs))`. Without them, the pairs are drawn from the graph itself:
positives are entailed pairs that are not direct edges, negatives are corrupted pairs outside the transitive closure.

### Checking a trained model

```python
model.containment()               # share of training edges whose child region is inside the parent region
model.transitivity_violations()   # share of predicted chains a<b<c with a<c not predicted (0 for the default rule)
model.evaluate(pos_pairs, neg_pairs)   # precision / recall / F1 at model.tau
model.rank(pos_pairs)                  # MRR / Hits@k of the true parent among all nodes
```

If `containment()` is low, the regions do not nest and `is_a` works like a nearest-neighbour test; lower `lam` or
the dimension.

## Presets

A preset fixes the training recipe; every option can still be set on top of it, e.g.
`RegD(dim=10, preset="strong-bce", lam=0, epochs=200)`.

| preset | use it for | loss | negatives | decision rule |
|---|---|---|---|---|
| `strong` (default) | embedding a given hierarchy | the paper's loss, margins 0.01 | re-drawn every epoch; never a pair from the transitive closure of the training edges | `b <= tau`, `tau <= 0` (transitive) |
| `strong-bce` | the same; better on tree-like graphs | BCE, temperature 0.1 | as above | as above |
| `predict` | incomplete graphs whose missing edges are to be predicted | BCE | re-drawn every epoch; only the training edges themselves are excluded | as above |
| `paper` | reproducing the paper | the paper's loss, margins 0 / 0.001 | drawn once, unfiltered | energy `b + lam d <= tau`, free `tau` |

## Options

`regd options` (or `regd.options_table()`) prints this list.

Regions

| option | default | choices | meaning |
|---|---|---|---|
| `dim` | 5 | | dimension of the space; a node has `2 * dim` numbers |
| `region` | `box` | `box`, `ball` | shape of a node's region |
| `boundary` | `distance` | `distance`, `volume` | `b` as the signed distance to the parent's boundary, or (boxes) the paper's "hybrid" log-volume of the part of the child outside the parent |

Energy and decision

| option | default | choices | meaning |
|---|---|---|---|
| `energy` | `paper` | `paper`, `poly`, `exp` | `b + lam d`; `(1+b)(1+lam d) - 1`; `e^b (1+lam d) - 1` |
| `depth` | `ratio` | `ratio`, `logsqrt`, `l1` | depth dissimilarity `d`: the paper's ratio, the metric `log(1 + sqrt(q))`, plain L1 |
| `lam` | 0.5 up to dim 5, else 0 | | weight of the depth term |
| `p` | 1 (box), 2 (ball) | 1, 2 | norm in the paper's `d` |
| `rule` | `transitive` | `transitive`, `tuned` | accept when score `<= tau` with `tau <= 0` (score = `b` for the paper energy, `E` for `poly` / `exp`); or the paper's free threshold on `b + lam d` |

Loss

| option | default | choices | meaning |
|---|---|---|---|
| `loss` | `cross` | `cross`, `bce`, `hinge` | the paper's contrastive loss; BCE; hinge |
| `margin` | 0.01 | | negatives are pushed to `E >= margin` (`cross`, `hinge`) |
| `pos_margin` | 0.01 | | positives are pushed to `E <= -pos_margin` |
| `temp` | 0.1 | | temperature of `bce` |

Negatives

| option | default | choices | meaning |
|---|---|---|---|
| `negatives` | `fresh` | `fresh`, `fixed` | re-drawn every epoch, or drawn once |
| `neg_filter` | `closure` | `closure`, `told`, `none` | pairs never used as negatives: the transitive closure of the training edges, only the edges, or nothing |
| `neg_side` | `both` | `both`, `parent`, `child` | which side of an edge is corrupted |
| `n_neg` | 9 | | negatives per edge and side |
| `hard_frac` | 0 | | share of hard negatives (wrong nodes with the lowest current energy) |
| `hard_pool` | 256 | | random candidate pool for hard negatives |

Optimisation and initialisation

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

**Which filter?** `closure` is right when the graph is the hierarchy you want to embed: every entailed pair is known,
so none of them should be pushed apart. If you train on an incomplete graph and want to predict missing edges, use
`preset="predict"` (`neg_filter="told"`); the closure of an incomplete graph says nothing about the missing part.

**Text initialisation.** `RegD(init="text")` encodes the node names once with a frozen sentence-transformers model,
projects the vectors to the table size with a fixed random matrix and normalises the rows. Training is unchanged and
the table is free afterwards. `fit(..., texts={node: "text"})` supplies other texts than the names;
`fit(..., init_vectors={node: vector})` supplies precomputed vectors and needs no encoder.

**The transitive rule and `boundary="volume"`.** The volume boundary is never negative, so with the transitive rule
`tau` can only be 0. Use it with `rule="tuned"` (as `preset="paper"` does).

## Results

All numbers: the paper's splits and stored negatives, test F1 (%) of the decision rule, CPU runs of this package
(records in `runs/*.jsonl`, commands from `scripts/paper_grid.py`). Each configuration uses the paper's learning rate, batch
size, epochs and `lam`; mammal 3 seeds, the others 2.

| data, region, dim | paper (Table 2) | `paper` | `strong` | `strong-bce` |
|---|---|---|---|---|
| mammal, box, 2 | 64.9 | 65.7 | 90.8 | **95.9** |
| mammal, box, 5 | 71.6 | 70.5 | 93.7 | **96.7** |
| mammal, ball, 2 | 62.7 | 55.8 | 88.8 | **95.4** |
| mammal, ball, 5 | 71.8 | 65.6 | 95.0 | **98.6** |
| noun, box, 5 | 53.8 | 52.5 | 50.7 | **54.6** |
| noun, box, 10 | 51.3 | 49.4 | 48.6 | **53.9** |
| noun, ball, 5 | 58.4 | 54.2 | 50.0 | **59.7** |
| noun, ball, 10 | 59.1 | 60.8 | 54.9 | **62.7** |
| MCG, box, 5 | 50.7 | 41.7 | **71.5** | 62.7 |
| MCG, box, 10 | 58.5 | 58.8 | **72.5** | 61.0 |
| MCG, ball, 5 | 44.9 | 45.2 | 71.1 | **74.6** |
| MCG, ball, 10 | 46.8 | 46.7 | 76.4 | **78.0** |
| Hearst, box, 5 | 42.8 | 43.7 | **54.3** | 47.4 |
| Hearst, box, 10 | 49.6 | 49.2 | **60.2** | 51.1 |
| Hearst, ball, 5 | 37.7 | 37.6 | **52.7** | 46.4 |
| Hearst, ball, 10 | 37.7 | 37.7 | **58.6** | 58.1 |
| mean | 53.9 | 52.2 | 68.1 | 68.5 |

- `paper` follows the published protocol and lands within about one point of Table 2 in 11 of 16 rows. The larger
  gaps: the two mammal balls and noun ball 5 (single published runs; the authors' own 10-seed means are 56.7, 66.8
  and 51.6) and MCG box 5, whose published setting uses a learned denominator that this package does not have.
- With `strong` every predicted relation is transitive (0 violations in all runs). With `paper` between 0% and 45%
  of the predicted chains `a < b < c` lack `a < c`.
- `strong-bce` is the better choice on the two WordNet graphs, `strong` on the two extracted graphs (MCG, Hearst);
  taking the better of the two per data type gives a mean of 70.9. `strong` is the default because it loses less
  where it is not the best.
- On noun the gain over the paper's protocol is small. That graph is trained for few steps per node (82,114 nodes).

One option at a time, mammal, box, 5 dimensions, 3 seeds (`runs/options.jsonl`); the reference rows are 93.7
(`strong`) and 96.7 (`strong-bce`):

| change | `strong` | `strong-bce` | comment |
|---|---|---|---|
| `neg_filter="told"` | 71.2 | 80.5 | no closure information |
| `neg_filter="none"` | 73.0 | 77.6 | |
| `negatives="fixed"` | 79.6 | 81.4 | closure-filtered but drawn once |
| `hard_frac=0.5` | 98.7 | 95.2 | 99.2 with all nodes as the pool; hurts on MCG / Hearst (71.5 -> 47.3, 52.7 -> 36.9) |
| `schedule="cosine"` | 97.2 | 92.3 | hurts on MCG (63.2) |
| `rule="tuned"` | 96.6 | 96.5 | 0.7% / 2.2% transitivity violations |
| `neg_side="child"` / `"parent"` | 96.1 / 54.9 | 97.0 / 63.2 | parent-only negatives leave reversed pairs unpunished |
| `n_neg` 4 / 19 / 49 | 95.5 / 95.1 / 93.5 | 96.8 / 96.9 / 96.8 | |
| `lam` 0 / 1 | 90.4 / 91.8 | 96.4 / 95.8 | |
| `energy="poly"` / `"exp"`, `depth="l1"` | 93.7 / 94.6 | 96.3 / 95.0 | |
| `energy="poly"` / `"exp"`, `depth="logsqrt"` | 93.3 / 94.1 | 96.2 / 94.4 | |
| `loss="hinge"`, margins 0.1 / 0.1 | 91.6 | | 94.7 with margin 0.5, positive margin 0 |
| `init="text"` | 94.0 | 97.6 | |
| ball instead of box | 95.0 | 98.6 | |

Dimension and `lam` (mammal, box; F1 and share of contained training edges):

| dim | `strong`, `lam=0` | `strong`, `lam=0.5` | `strong-bce`, `lam=0` | `strong-bce`, `lam=0.5` |
|---|---|---|---|---|
| 2 | 75.3 / 0.97 | 90.1 / 0.98 | 88.5 / 0.99 | 91.2 / 1.00 |
| 5 | 90.4 / 0.98 | 93.7 / 0.97 | 96.4 / 1.00 | 96.7 / 1.00 |
| 10 | 94.2 / 0.98 | 95.5 / 0.96 | 97.8 / 1.00 | 98.3 / 1.00 |
| 20 | 95.1 / 0.97 | 95.7 / 0.93 | 98.5 / 1.00 | 97.6 / 1.00 |
| 50 | 97.8 / 0.98 | 95.3 / 0.85 | 97.9 / 0.99 | 97.6 / 0.95 |

The depth term matters most in very low dimensions and costs containment in high ones, hence the default
(`lam = 0.5` up to 5 dimensions, 0 above).

Text initialisation (`init="text"`, BGE-base), 5 dimensions unless noted, 2 seeds:

| data | preset | random init | text init |
|---|---|---|---|
| noun, box, 5 / 10 | `strong-bce` | 54.6 / 53.9 | 61.1 / 64.2 |
| noun, ball, 5 / 10 | `strong-bce` | 59.7 / 62.7 | 64.1 / 62.2 |
| MCG, box / ball | `strong` | 71.5 / 71.1 | 69.7 / 70.2 |
| Hearst, box / ball | `strong` | 54.3 / 52.7 | 52.5 / 50.3 |
| mammal, box | `strong` / `strong-bce` | 93.7 / 96.7 | 94.0 / 97.6 |

Text initialisation helps on the large WordNet graph, makes no clear difference on mammal and costs one or two points
on MCG and Hearst.

What these numbers do and do not show:

- The closure filter uses only the training edges, but every test pair of these benchmarks is entailed by them (a
  purely symbolic closure lookup scores F1 = 1). The numbers measure how well the regions reproduce a known hierarchy.
- They do not measure the prediction of unknown edges. With 10% of the mammal edges hidden and `neg_filter="told"`,
  the hidden parent is ranked poorly: MRR 0.03 (Hits@10 0.05) with random initialisation and 0.05 (0.13) with
  `init="text"`, 3 seeds (`scripts/predict_check.py`). Most hidden edges belong to leaves that then occur in no
  training edge; a free table cannot place such a node.

Time on 4 CPU threads: mammal 20-45 s; MCG and Hearst 5-8 min; noun 6-8 min (13 min with text initialisation,
including the encoding).

Reproduce: `python scripts/bench.py data_utils/data/maxn/mammal_closure.tsv --preset strong --seeds 43 0 1`, or all
of it with `python scripts/paper_grid.py data_utils/data > jobs.txt` (the data are those of the
[paper code](#dataset-preparation)).

## Relation to the paper

Same as the paper: the regions (boxes with centre and offsets, balls with centre and radius), the boundary distance
`b`, the depth dissimilarity `d`, the energy `b + lam d` with the depth term switched off in training for contained
pairs, the contrastive loss, the initialisation, Adam. `preset="paper"` follows the published protocol.

Training changes (model and energy unchanged):

- negatives re-drawn every epoch instead of once;
- negatives never taken from the transitive closure of the training edges (`closure`) or from the edges (`told`);
- margins 0.01 on both sides; optional BCE loss; optional hard negatives;
- every edge is used in every epoch (the original loader dropped the last incomplete batch of edges).

New, not in the paper:

- **Transitive decision rule** (default): accept when `b <= tau` with `tau <= 0`. The depth term then only shapes
  training. The paper's rule, a free threshold on `b + lam d`, is not transitive.
- **Multiplicative energies** `poly` and `exp`, decided by `E <= tau <= 0`, in which the depth term takes part in
  the decision and transitivity is kept when the depth is a metric (`depth="logsqrt"`).
- **`lam` default by dimension.** Above about 10 dimensions the depth term makes boxes stop nesting, so the default
  is 0 there.
- **Text initialisation**, the hinge loss, one-sided negatives, the cosine schedule and early stopping.
- The query / save / load / export / plot interface and the command line.

Left out of the package (available in the [paper code](#paper-code)): the learned denominator (`add_denominator`), the
"hybrid" variant for balls, the ontology plug-in (`EXP_ontology`) and the baselines.

The test `test_same_numbers_as_original_code` checks that `b`, `d`, the training and evaluation energies and the loss
agree numerically with the original `BoxDistBoundary.py` at the root of this repository.

## Tests

```bash
pip install ".[test]" && pytest
```

## Paper code

The scripts of the paper are kept as they were. They are based on the
[ShadowCones](https://github.com/ydtydr/ShadowCones) project and do not need the `regd` package.

### Illustration


The embeddings of the original code are illustrated in `mammal_dim=2_ball.png` (top of this page). In this figure, the x-axis represents the sum of the coordinates of each center (i.e., *x* + *y* for a center located at (*x*, *y*)), while the y-axis represents the corresponding radius. As shown in the figure, our embeddings effectively capture the hierarchical structure of the mammal dataset.



### Installation

#### Requirements
- Python >= 3.9
- PyTorch >= 1.12.1

#### Setup
 
```
# Create a new conda environment
conda create -n myenv python=3.9.20

# Activate the environment
conda activate myenv

# Install packages with specific versions
pip install \
    torch==2.5.0 \
    torchaudio==2.5.0 \
    torchvision==0.20.0 \
    six==1.16.0 \
    tqdm==4.66.5 \
    scikit-learn==1.5.2 \
    numpy==1.26.4 \
    gensim==4.3.3

```

#### Dataset Preparation
1. First, unzip the `data_utils.zip` file in the project root directory.
2. Download the MCG and Hearst datasets using the following commands:
```bash
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1WH2LIk2EsTe_lQ03AjCaxZ3o8fSkNt1f?usp=sharing
```
3. Move the downloaded datasets into the `data_utils` directory.

### Usage

#### Training
To train the model, use the following command:
```bash
python train.py -dataset mammal -dim 2 -epoch 400 -lr 0.01 -mode box -hybrid 0 -lamda 1.0
```


#### Parameters
- `dataset`: Name of the dataset to use (e.g., "mammal", "MCG", "hearst")
- `mode`: Type of region to use for embedding:
  - "box": Axis-aligned boxes
  - "ball": Spherical regions
- `hybrid`: Boundary distance calculation method:
  - 0: Use our boundary distance
  - 1: Use specific boundary distance for boxes/balls
- `lamda`: Weight for the depth distance in the energy function (default: 1.0)
- `dim`: Dimension of the embedding space (default: 2)
- `epoch`: Number of training epochs (default: 400)
- `lr`: Learning rate (default: 0.01)

For recommended hyperparameter settings and example commands, refer to `run.sh`.
