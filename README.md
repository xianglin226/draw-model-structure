# draw-model-structure

Publication-ready architecture diagrams of neural-network models, rendered
from a small Python *spec file* with [Graphviz](https://graphviz.org/).
Outputs editable PNG / SVG / PDF at multiple levels of abstraction
(`--depth 1 → 4+`), with semantic colour coding, dataflow arrows, and an
auto-generated legend.

The renderer is model-agnostic: you describe your architecture as a
hierarchy of `Block`s (and optionally `Matrix` blocks for tensor-shaped
inputs/outputs), and it draws the figure for you.

---

## Features

- **Multiple abstraction levels** from one spec — re-render at any
  `--depth` to peel one layer at a time (root → children → grandchildren →
  …).
- **Editable vector output.** SVG opens cleanly in Adobe Illustrator,
  Inkscape, or Photoshop with one `<g>` per box, gradient fills, and
  selectable text.
- **Semantic colour coding.** Each block has a `kind` (e.g. `attn`, `mlp`,
  `embedding`, `io`); leaf colours are looked up from a palette and the
  legend documents what each colour means.
- **Built-in colour schemes:** `default`, `dark`, `colorful`, `minimal`.
- **Matrix-shaped I/O.** `Matrix` renders a leaf as a small heatmap grid
  with row/column group labels — perfect for showing what tensors a model
  consumes and produces, including missing-value masks.
- **Deterministic rendering** (matrix patterns are seeded), so the same
  spec produces the same figure every time.
- **Tree-print mode** (`--mode text`) for a quick terminal sanity check.

---

## Installation

### Requirements

- Python 3.9+ (no third-party Python packages needed — only the standard
  library).
- The Graphviz `dot` binary on `PATH`.

Install Graphviz:

```bash
# macOS
brew install graphviz

# Debian / Ubuntu
sudo apt-get install graphviz

# Fedora / RHEL
sudo dnf install graphviz

# conda
conda install -c conda-forge graphviz
```

Verify:

```bash
dot -V
```

### Get the code

```bash
git clone <this-repo-url> draw-model-structure
cd draw-model-structure
```

No `pip install` step is required.

---

## Quick start

Render the bundled Transformer example at three depths:

```bash
mkdir -p plots

python scripts/render_model.py --spec examples/transformer_spec.py \
    --depth 1 --out plots/transformer_d1.svg

python scripts/render_model.py --spec examples/transformer_spec.py \
    --depth 2 --out plots/transformer_d2.svg

python scripts/render_model.py --spec examples/transformer_spec.py \
    --depth 3 --out plots/transformer_d3.svg
```

Render the matrix-I/O example (set-attention imputer with support/query
row groups and a missingness mask):

```bash
python scripts/render_model.py --spec examples/matrix_io_spec.py \
    --depth 3 --out plots/matrix_io_d3.svg
```

Open the resulting SVG/PNG in your viewer, or load the SVG into
Illustrator/Inkscape for further editing.

---

## Writing a spec file

A spec is a plain Python module that exports a top-level variable named
`SPEC` (a `Block` instance). Optionally it can export `LEGEND_ENTRIES` and
`PALETTE` as well.

The spec file is free to do `from render_model import Block, Matrix` —
the renderer adds its own directory to `sys.path` before importing the
spec, so no path gymnastics are needed.

### Minimal example

```python
# my_model_spec.py
from render_model import Block

SPEC = Block("MyTransformer", kind="model", children=[
    Block("Inputs\ntokens · positions · masks", kind="io"),
    Block("Encoder x N", kind="encoder_layer", children=[
        Block("Multi-Head Attention\nQ - K - V", kind="attn"),
        Block("MLP + LayerNorms\nfeed-forward block", kind="mlp"),
    ]),
    Block("Output Head\nlogits", kind="head"),
    Block("Outputs\npredicted tokens", kind="io"),
])
```

### `Block` fields

| Field | Meaning |
|---|---|
| `name` | Display label. Text before the first `\n` becomes the **title** rendered above the box (larger, bold). Text after `\n` becomes the **subtitle** rendered inside the box (smaller). |
| `kind` | Semantic category used to look up the leaf-box fill colour. Container blocks (those with `children`) ignore their kind — container fill is always neutral. |
| `children` | Sub-blocks. When the depth makes them visible they are stacked vertically inside this block; otherwise this block is rendered as a single leaf using its description. |
| `flow` (default `True`) | If `True`, a small `↓` arrow is drawn between consecutive children. Set `False` for groups of parallel siblings. |
| `flatten_when_expanded` (default `False`) | If `True`, this wrapper block disappears once its children become visible — the children take its slot in the parent's chain. Useful for a single "Backbone" block that exists only at depth 1. |

### Built-in `kind` palette

- `io` — inputs, outputs (light blue)
- `embedding`, `emb` — embeddings (purple)
- `proj`, `head`, `decoder` — projections / output heads (peach)
- `attn`, `feat_attn` — attention (red / pink)
- `sample_attn` — sample/set attention (blue)
- `mlp` — feed-forward / MLP (green)
- `encoder`, `encoder_layer` — encoder containers (orange; only visible when rendered as a leaf at a coarse depth)
- `model` — root container (white)

Add your own kinds via the optional `PALETTE` export:

```python
PALETTE = {
    "router":    "#FFD27F",
    "expert":    "#A0E7E5",
}
```

`PALETTE` is merged on top of the chosen scheme, so it can also override
existing kinds.

### Custom legend

```python
LEGEND_ENTRIES = [
    ("io",     "Input / Output"),
    ("attn",   "Attention"),
    ("mlp",    "Feed-forward"),
    ("router", "MoE Router"),
    ("expert", "MoE Expert"),
]
```

If omitted, a default six-entry legend is used. Pass `--no_legend` on the
CLI to suppress the legend entirely.

---

## Matrix-shaped blocks (heatmap I/O)

For data-tensor visualisation, use the `Matrix` constructor instead of a
plain `Block`. It renders the leaf as a small heatmap grid with row/column
group labels — useful for showing what a tensor input/output looks like.

### Flat layout (single row group)

```python
from render_model import Block, Matrix

Matrix("Inputs\nN samples x features", kind="io", rows=5, groups=[
    ("modality A", 10, 0.00, "#9CB3E5"),
    ("modality B",  8, 0.50, "#88B04B"),
])
```

Each `groups` entry is `(label, ncols, missing_frac, color)`:

- `label` — text shown below the matrix for that column group; pass `""`
  to hide.
- `ncols` — number of columns this group occupies.
- `missing_frac` — fraction of cells in this group rendered in light grey
  to indicate missingness. `0.0` = fully observed.
- `color` — base colour for cells in this group; pass `None` to fall back
  to the block's `kind` colour.

### Row groups (rows with different roles)

When rows have different roles — e.g. support vs query, train vs test,
masked vs unmasked — pass `row_groups` instead. Each row group has its
own label, row count, and column-group list:

```python
Matrix("Inputs\nN samples x (A + B)", kind="io", row_groups=[
    ("support", 3, [
        ("modality A", 10, 0.00, "#9CB3E5"),
        ("modality B",  8, 0.20, "#88B04B"),
    ]),
    ("query", 2, [
        ("modality A", 10, 0.00, "#9CB3E5"),
        ("modality B",  8, 0.80, "#88B04B"),
    ]),
])
```

All row groups within a matrix should share the same column structure
(column counts and labels); only `missing_frac` and `color` should vary
per group.

### Transposing

Pass `transposed=True` to flip rows ↔ columns at render time:

```python
Matrix("Inputs", kind="io", transposed=True, row_groups=[...])
```

Use this when the feature axis is much longer than the sample axis, or
when long column labels are stretching the figure horizontally.

The rendering is **deterministic** (seeded with `seed`, default 42), so
the same spec always produces the same pattern of cell shades and missing
cells.

---

## CLI reference

```text
python scripts/render_model.py --spec <path/to/spec.py> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--spec PATH` | *(required)* | Path to a Python spec file exporting `SPEC`. |
| `--depth N` | `2` | Conceptual depth: 1 = root only; 2 = direct children; 3+ = deeper. |
| `--out PATH` | `plots/model_architecture.png` | Output path. Format inferred from suffix: `.png`, `.svg`, `.pdf`, or `.dot`. |
| `--mode {dot,text}` | `dot` | `dot` = graphviz diagram; `text` = ASCII tree printed to stdout. |
| `--scheme {default,dark,colorful,minimal}` | `default` | Colour palette. |
| `--rankdir {LR,TB}` | `LR` | Layout direction across modules. `LR` = horizontal, `TB` = vertical stack. |
| `--no_intra_arrows` | off | Suppress the small `↓` flow arrows drawn between sub-blocks inside each module. |
| `--no_legend` | off | Suppress the colour legend at the bottom. |

### Depth cheat sheet

| `--depth` | Shows |
|---|---|
| `1` | One big root block. |
| `2` | Direct children of the root. |
| `3` | Grandchildren. |
| `4+` | Deeper. Stop when it stops being useful — usually ≤ 4. |

---

## Design conventions

When defining a spec, follow these conventions so the figure reads well:

- **Short titles + subtitles.** Put a brief name first, then `\n`, then
  a short description. The renderer word-wraps both automatically (titles
  ≤ 30 chars/line, descriptions ≤ 40 chars/line, balanced so two-line
  wraps split near the middle).
- **Consistent `kind` semantics.** Two blocks with the same `kind` should
  mean the same thing — that's what the legend promises the reader.
- **Use `kind="io"` for top-level inputs/outputs.** This guarantees the
  conventional light-blue I/O boxes at the ends of the flow.
- **Mark virtual wrappers with `flatten_when_expanded=True`.** Avoids a
  redundant outer box once you expand into its children.
- **Stay shallow.** Aim for depth ≤ 4. Deeper trees usually mean you
  should consolidate at a higher abstraction level.
- **Hand-curate, don't auto-introspect.** PyTorch module trees are too
  noisy and parameter-heavy to render directly; write the spec at the
  level of conceptual blocks (Attention, MLP, Embedding) rather than the
  level of `nn.Linear`, `nn.Dropout`, `nn.LayerNorm`.

## What the renderer guarantees

- **Title above each box** (larger, bold) using a fixed per-level font
  cascade (18 → 14 → 12 → 11 pt).
- **Subtitle inside the box** (smaller) at a uniform 10 pt across the
  whole figure, regardless of nesting depth — so identical-level boxes
  look identical across modules.
- **Containers are neutral** (white in light schemes, dark in
  `--scheme dark`); only leaf boxes carry semantic colour.
- **Within-module flow arrows** (`↓`) between consecutive children, plus
  cross-module arrows attached to the midpoint of each box's border.
- **A legend** at the bottom mapping every leaf colour to its meaning.

---

## Anti-patterns

- **Auto-introspecting an `nn.Module`.** The resulting tree is too
  granular (one block per `nn.Linear`); the figure becomes unreadable.
  Hand-curate the spec at the level of conceptual blocks.
- **Long single-line `name` strings.** They make boxes wide. Always put
  a `\n` between the short title and its description.
- **Reusing one `kind` for unrelated blocks.** The legend will mislead
  the reader. Add a new kind via `PALETTE` and `LEGEND_ENTRIES`.
- **Depth > 5.** Almost always means the spec is over-decomposed. Roll
  lower-level blocks back into their parents.

---

## Repository layout

```
draw-model-structure/
├── README.md
├── scripts/
│   └── render_model.py        # Generic renderer + CLI; exports Block and Matrix
└── examples/
    ├── transformer_spec.py    # Minimal Transformer example
    └── matrix_io_spec.py      # Matrix I/O + row groups + nested encoder/decoder
```

---

## License

MIT.
