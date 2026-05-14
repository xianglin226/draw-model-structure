"""Example spec demonstrating ``Matrix`` I/O blocks.

Showcases:

- Two modalities at the input, one with missing values.
- Row groups (support vs query) with different missingness fractions,
  to depict an in-context-learning / set-attention style of model.
- Transposed matrices so the feature axis is vertical.
- A nested encoder/decoder backbone with multiple ``kind`` colours.

Render with::

    python scripts/render_model.py --spec examples/matrix_io_spec.py \
        --depth 3 --out plots/matrix_io_d3.svg
"""

from render_model import Block, Matrix


# Per-modality cell colours used by the Matrix blocks.
_MOD_A = "#9CB3E5"  # blue
_MOD_B = "#88B04B"  # green


SPEC = Block("Set-Attention Imputer", kind="model", children=[
    Matrix("Inputs\nN samples x (modality A + modality B)", kind="io",
           transposed=True, row_groups=[
        ("support", 5, [
            ("modality A",                      10, 0.00, _MOD_A),
            ("modality B with missing values",   8, 0.20, _MOD_B),
        ]),
        ("query", 2, [
            ("modality A",                      10, 0.00, _MOD_A),
            ("modality B with missing values",   8, 0.80, _MOD_B),
        ]),
    ]),
    Block("Backbone\nself-supervised encoder",
          kind="encoder", flatten_when_expanded=True, children=[
        Block("Token Embedding\ninput projections", kind="embedding",
              children=[
            Block("Modality A Embedding\nvalue MLP", kind="proj"),
            Block("Modality B Embedding\nvalue MLP + mask token",
                  kind="proj"),
            Block("Meta Embedding\ncovariates / context", kind="emb"),
        ]),
        Block("Encoder x N\nstacked attention blocks",
              kind="encoder_layer", children=[
            Block("Feature Attention\nfeature-level mixing",
                  kind="feat_attn"),
            Block("Sample Attention\ncross-sample, symmetric",
                  kind="sample_attn"),
            Block("MLP + LayerNorms\nfeed-forward", kind="mlp"),
        ]),
        Block("Decoder Head\nimpute missing values", kind="decoder",
              children=[
            Block("Cross-Attention\nquery against support", kind="attn"),
            Block("Output MLP\nproject to predictions", kind="mlp"),
            Block("Prediction Head\nper-feature output", kind="head"),
        ]),
    ]),
    Matrix("Outputs\nN_query x modality B", kind="io",
           transposed=True, row_groups=[
        ("query", 2, [
            ("modality B (imputed)", 8, 0.0, _MOD_B),
        ]),
    ]),
])


LEGEND_ENTRIES = [
    ("io",          "Input / Output"),
    ("emb",         "Embedding"),
    ("proj",        "Projection / Head"),
    ("attn",        "Attention"),
    ("sample_attn", "Sample Attention"),
    ("mlp",         "MLP / Feed-forward"),
]
