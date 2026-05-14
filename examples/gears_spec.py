"""Example spec: GEARS — graph-enhanced perturbation response model.

GEARS predicts post-perturbation gene expression from a control profile
plus a list of perturbation IDs. Two graph neural networks supply the
inductive bias:

- a GNN over a gene **co-expression** graph produces a per-gene position
  embedding that augments a learned per-gene base embedding;
- a GNN over a perturbation **GO-similarity** graph produces a per-
  perturbation embedding that is added to the rows of the gene-embedding
  matrix corresponding to the perturbed genes.

A shared MLP plus two gene-specific weight tensors decode the fused
representation; the control expression is added back at the end as a
residual.

Render with::

    python scripts/render_model.py --spec examples/gears_spec.py \
        --depth 2 --out plots/gears_d2.svg
    python scripts/render_model.py --spec examples/gears_spec.py \
        --depth 3 --out plots/gears_d3.svg
"""

from render_model import Block, Matrix


_EXP_COLOR  = "#9CB3E5"  # blue   — gene expression values
_PERT_COLOR = "#F5B7B1"  # pink   — perturbation index values


SPEC = Block("GEARS", kind="model", children=[
    Matrix("Inputs\nN cells x n_genes (control) + perturbation IDs",
           kind="io", transposed=True, row_groups=[
        ("perturbed cells", 4, [
            ("control gene expression", 14, 0.0, _EXP_COLOR),
            ("perturbation IDs",         2, 0.0, _PERT_COLOR),
        ]),
    ]),
    Block("Encoder\ndual graph encoder + perturbation fusion",
          kind="encoder", flow=False, children=[
        Block("Gene Embedding\nbase + co-expression position",
              kind="embedding", children=[
            Block("Gene Lookup\nlearned per-gene vector", kind="emb"),
            Block("Position GNN x L_g\nSGConv over co-expression graph",
                  kind="gnn"),
            Block("Fusion MLP\nbase + 0.2 . position", kind="mlp"),
        ]),
        Block("Perturbation Embedding\nGO-graph perturbation context",
              kind="embedding", children=[
            Block("Perturbation Lookup\nlearned per-perturbation vector",
                  kind="emb"),
            Block("Perturbation GNN x L_p\nSGConv over GO-similarity graph",
                  kind="gnn"),
        ]),
        Block("Perturbation Fusion\nadd pert vectors to perturbed-gene rows",
              kind="proj", children=[
            Block("Per-cell Pert MLP\npert_fuse projection", kind="mlp"),
            Block("Row-indexed Add\ninto perturbed gene rows", kind="proj"),
        ]),
    ]),
    Block("Decoder\nshared MLP + gene-specific decoding",
          kind="decoder", children=[
        Block("Shared Recovery MLP\nhidden -> hidden", kind="mlp"),
        Block("Gene-Specific Weight 1\nindv_w1 + indv_b1", kind="head"),
        Block("Cross-Gene MLP\ncross_gene_state(num_genes -> hidden)",
              kind="mlp"),
        Block("Gene-Specific Weight 2\nindv_w2 + indv_b2", kind="head"),
        Block("Residual Add\noutput += control expression", kind="proj"),
    ]),
    Matrix("Outputs\nN cells x n_genes (post-perturbation expression)",
           kind="io", transposed=True, row_groups=[
        ("perturbed cells", 4, [
            ("predicted post-pert expression", 14, 0.0, _EXP_COLOR),
        ]),
    ]),
])


PALETTE = {
    "gnn": "#FFD27F",
}


LEGEND_ENTRIES = [
    ("io",   "Input / Output"),
    ("emb",  "Embedding lookup"),
    ("gnn",  "Graph Neural Network"),
    ("mlp",  "MLP / Feed-forward"),
    ("proj", "Projection / Fusion"),
    ("head", "Output Head"),
]
