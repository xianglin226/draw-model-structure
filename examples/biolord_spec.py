"""Example spec: biolord — disentangled attribute-aware autoencoder.

biolord encodes single-cell gene expression into a *composite* latent
that disentangles per-sample "unknown" variation from a set of known
attributes. Each known attribute has its own encoder:

- **categorical** attributes (e.g. perturbation, tissue, donor) are
  embedded with one ``nn.Embedding`` per attribute;
- **ordered** / continuous attributes (e.g. dose, time) are embedded
  with one ``FCLayers`` MLP per attribute;
- the **unknown attribute** factor is a per-sample ``RegularizedEmbedding``
  that adds Gaussian noise during training.

All latent vectors are concatenated and decoded back to a gene
expression distribution: Gaussian (with mean / variance heads) for
``gene_likelihood="normal"``, or a Negative Binomial / Poisson rate via
scvi's ``DecoderSCVI`` otherwise.

Render with::

    python scripts/render_model.py --spec examples/biolord_spec.py \
        --depth 2 --out plots/biolord_d2.svg
    python scripts/render_model.py --spec examples/biolord_spec.py \
        --depth 3 --out plots/biolord_d3.svg
"""

from render_model import Block, Matrix


_EXP_COLOR  = "#9CB3E5"  # blue   — gene expression
_ATTR_COLOR = "#88B04B"  # green  — attribute values


SPEC = Block("biolord", kind="model", children=[
    Matrix("Inputs\nper-cell expression + known attributes",
           kind="io", transposed=True, row_groups=[
        ("cells", 6, [
            ("gene expression",          12, 0.0, _EXP_COLOR),
            ("attributes (cat. + ord.)",  6, 0.0, _ATTR_COLOR),
        ]),
    ]),
    Block("Latent Encoder\ndisentangled attribute-aware latent",
          kind="encoder", flow=False, children=[
        Block("Sample Latent\nunknown-attribute factor",
              kind="embedding", children=[
            Block("Per-Sample Lookup\nRegularizedEmbedding(sample_idx)",
                  kind="emb"),
            Block("Gaussian Noise (train)\nN(0, sigma^2) regulariser",
                  kind="proj"),
        ]),
        Block("Categorical Attribute Encoders\nperturbation, tissue, donor, ...",
              kind="embedding", children=[
            Block("nn.Embedding x K_cat\none per categorical attribute",
                  kind="emb"),
        ]),
        Block("Ordered Attribute Encoders\ndose, time, continuous covariates",
              kind="embedding", children=[
            Block("FCLayers x K_ord\nMLP per ordered attribute",
                  kind="mlp"),
        ]),
        Block("Latent Concatenation\nz = [sample | cat | ord]",
              kind="proj"),
    ]),
    Block("Decoder\ngene expression generator",
          kind="decoder", children=[
        Block("Decoder MLP\nFCLayers stack (n_in -> n_genes)",
              kind="mlp"),
        Block("Likelihood Head\nGaussian (mean, var) | NB / Poisson rate",
              kind="head"),
    ]),
    Matrix("Outputs\nN cells x n_genes (predicted distribution)",
           kind="io", transposed=True, row_groups=[
        ("cells", 6, [
            ("predicted expression mean", 12, 0.0, _EXP_COLOR),
        ]),
    ]),
])


LEGEND_ENTRIES = [
    ("io",   "Input / Output"),
    ("emb",  "Embedding lookup"),
    ("mlp",  "MLP / Feed-forward"),
    ("proj", "Projection / Fusion"),
    ("head", "Output Head"),
]
