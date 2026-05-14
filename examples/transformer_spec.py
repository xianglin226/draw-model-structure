"""Example spec: a generic Transformer encoder.

This is a minimal, model-agnostic spec demonstrating the basic ``Block``
hierarchy. Copy and edit it as a starting point for your own model.

Render with::

    python scripts/render_model.py --spec examples/transformer_spec.py \
        --depth 2 --out plots/transformer_d2.svg
    python scripts/render_model.py --spec examples/transformer_spec.py \
        --depth 3 --out plots/transformer_d3.svg
"""

from render_model import Block


SPEC = Block("Transformer", kind="model", children=[
    Block("Inputs\ntokens · positions · masks", kind="io"),
    Block("Embedding\ntoken + positional", kind="embedding"),
    Block("Encoder x N\nN stacked attention layers",
          kind="encoder_layer", children=[
        Block("Multi-Head Self-Attention\nQ - K - V projections",
              kind="attn"),
        Block("MLP + LayerNorms\nfeed-forward block", kind="mlp"),
    ]),
    Block("Output Head\nvocab logits", kind="head"),
    Block("Outputs\nnext-token distribution", kind="io"),
])


LEGEND_ENTRIES = [
    ("io",   "Input / Output"),
    ("emb",  "Embedding"),
    ("attn", "Attention"),
    ("mlp",  "MLP / Feed-forward"),
    ("head", "Output Head"),
]
