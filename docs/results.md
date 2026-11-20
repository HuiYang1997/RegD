# Results

## The 16 configurations of the paper

Every run on this page sets `lam` explicitly (one training run per model).

All numbers: the paper's splits and stored negatives, test F1 (%) of the decision rule, CPU runs of this package
(records in `runs/*.jsonl`, commands from `scripts/paper_grid.py`; run from the repository root). Each configuration uses the paper's learning rate, batch
size, epochs and `lam`; mammal 3 seeds, the others 2.

| data, region, dim | paper (Table 2) | `strong` | `strong-bce` |
|---|---|---|---|
| mammal, box, 2 | 64.9 | 90.8 | **95.9** |
| mammal, box, 5 | 71.6 | 93.7 | **96.7** |
| mammal, ball, 2 | 62.7 | 88.8 | **95.4** |
| mammal, ball, 5 | 71.8 | 95.0 | **98.6** |
| noun, box, 5 | 53.8 | 50.7 | **54.6** |
| noun, box, 10 | 51.3 | 48.6 | **53.9** |
| noun, ball, 5 | 58.4 | 50.0 | **59.7** |
| noun, ball, 10 | 59.1 | 54.9 | **62.7** |
| MCG, box, 5 | 50.7 | **71.5** | 62.7 |
| MCG, box, 10 | 58.5 | **72.5** | 61.0 |
| MCG, ball, 5 | 44.9 | 71.1 | **74.6** |
| MCG, ball, 10 | 46.8 | 76.4 | **78.0** |
| Hearst, box, 5 | 42.8 | **54.3** | 47.4 |
| Hearst, box, 10 | 49.6 | **60.2** | 51.1 |
| Hearst, ball, 5 | 37.7 | **52.7** | 46.4 |
| Hearst, ball, 10 | 37.7 | **58.6** | 58.1 |
| mean | 53.9 | 68.1 | 68.5 |

- "paper (Table 2)" is the number published in the paper; the other two columns are runs of this package.
- With `strong` and `strong-bce` every predicted relation is transitive (0 violations in all runs).
- `strong-bce` is the better choice on the two WordNet graphs, `strong` on the two extracted graphs (MCG, Hearst);
  taking the better of the two per data type gives a mean of 70.9. `strong` is the default because it loses less
  where it is not the best.
- On noun the gain over the paper's numbers is small. That graph is trained for few steps per node (82,114 nodes).

## One option at a time

Mammal, box, 5 dimensions, 3 seeds (`runs/options.jsonl`); the reference rows are 93.7
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

## Dimension and `lam`

F1 and share of contained training edges (mammal, box):

| dim | `strong`, `lam=0` | `strong`, `lam=0.5` | `strong-bce`, `lam=0` | `strong-bce`, `lam=0.5` |
|---|---|---|---|---|
| 2 | 75.3 / 0.97 | 90.1 / 0.98 | 88.5 / 0.99 | 91.2 / 1.00 |
| 5 | 90.4 / 0.98 | 93.7 / 0.97 | 96.4 / 1.00 | 96.7 / 1.00 |
| 10 | 94.2 / 0.98 | 95.5 / 0.96 | 97.8 / 1.00 | 98.3 / 1.00 |
| 20 | 95.1 / 0.97 | 95.7 / 0.93 | 98.5 / 1.00 | 97.6 / 1.00 |
| 50 | 97.8 / 0.98 | 95.3 / 0.85 | 97.9 / 0.99 | 97.6 / 0.95 |

The depth term matters most in low dimensions and costs containment in high ones. The default is
`lam_set = [0.5]`. With `lam_set = [0, 0.5]` both values are trained and the one with the better validation F1 is
kept; the selection looks at F1 only, not at containment.

## Text initialisation

`init="text"`, BGE-base, 5 dimensions unless noted, 2 seeds:

| data | preset | random init | text init |
|---|---|---|---|
| noun, box, 5 / 10 | `strong-bce` | 54.6 / 53.9 | 61.1 / 64.2 |
| noun, ball, 5 / 10 | `strong-bce` | 59.7 / 62.7 | 64.1 / 62.2 |
| MCG, box / ball | `strong` | 71.5 / 71.1 | 69.7 / 70.2 |
| Hearst, box / ball | `strong` | 54.3 / 52.7 | 52.5 / 50.3 |
| mammal, box | `strong` / `strong-bce` | 93.7 / 96.7 | 94.0 / 97.6 |

Text initialisation helps on the large WordNet graph, makes no clear difference on mammal and costs one or two points
on MCG and Hearst.

## What these numbers do and do not show


- The closure filter uses only the training edges, but every test pair of these benchmarks is entailed by them (a
  purely symbolic closure lookup scores F1 = 1). The numbers measure how well the regions reproduce a known hierarchy.
- They do not measure the prediction of unknown edges. With 10% of the mammal edges hidden and `neg_filter="told"`,
  the hidden parent is ranked poorly: MRR 0.03 (Hits@10 0.05) with random initialisation and 0.05 (0.13) with
  `init="text"`, 3 seeds (`scripts/predict_check.py`). Most hidden edges belong to leaves that then occur in no
  training edge; a free table cannot place such a node.

## Running time and reproduction

Time on 4 CPU threads: mammal 20-45 s; MCG and Hearst 5-8 min; noun 6-8 min (13 min with text initialisation,
including the encoding).

Reproduce: `python scripts/bench.py paper/data_utils/data/maxn/mammal_closure.tsv --preset strong --seeds 43 0 1`, or all
of it with `python scripts/paper_grid.py paper/data_utils/data > jobs.txt` (the data are those of the
[paper code](../paper/README.md#dataset-preparation)).
