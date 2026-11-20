# Relation to the paper

Same as the paper: the regions (boxes with centre and offsets, balls with centre and radius), the boundary distance
`b`, the depth dissimilarity `d`, the energy `b + lam d` with the depth term switched off in training for contained
pairs, the contrastive loss, the initialisation, Adam. To reproduce the paper itself, use the
original code in [`paper/`](../paper/README.md).

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
- **`lam_set`.** The default weight of the depth term is 0.5 (the paper sets it per configuration: 1.0, 0.5 or
  0). Given several values, the package trains once per value and keeps the best on validation data.
- **Text initialisation**, the hinge loss, one-sided negatives, the cosine schedule and early stopping.
- The query / save / load / export / plot interface and the command line.

Left out of the package (available in the [paper code](../paper/README.md)): the learned denominator (`add_denominator`), the
"hybrid" variant for balls, the ontology plug-in (`EXP_ontology`) and the baselines.

The test `test_same_numbers_as_original_code` checks that `b`, `d`, the training and evaluation energies and the loss
agree numerically with the original `BoxDistBoundary.py` in `paper/`.
