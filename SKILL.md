---
name: draw-model-structure
description: Draw publication-ready architecture diagrams of neural-network models at multiple levels of abstraction (depth 1 → 4+) using Graphviz. Renders PNG/SVG with semantic colour coding, depth-controlled detail, dataflow arrows inside and between modules, and a colour legend. Use when the user asks to visualise a model architecture, diagram a network, draw model blocks, generate a paper-quality figure of a neural net, or compare the structure of two models.
---

# Draw Model Structure

## When to use this skill

Use this skill whenever the user asks to:

- Visualise / diagram / draw a model architecture.
- Generate a figure of a network for a paper, slide deck, or PR description.
- Render the same architecture at several levels of abstraction (e.g. one-block summary vs full encoder breakdown).
- Compare the structure of two model variants side by side.

The output is Graphviz-rendered PNG **and** editable SVG (opens cleanly in Adobe Illustrator, Inkscape, or Photoshop).

## Prerequisites

- Python 3.9+ (standard library only).
- The Graphviz `dot` binary on `PATH`. On Linux: `apt install graphviz` / `dnf install graphviz`. On macOS: `brew install graphviz`. On conda: `conda install -c conda-forge graphviz`. Verify with `dot -V`.

## High-level workflow

1. **Write a spec file** — a small Python module describing the architecture as a hierarchy of `Block`s.
2. **Render** — run `scripts/render_model.py --spec <spec.py>` to produce PNG / SVG / PDF.
3. **Iterate** — tweak depth, colour scheme, intra-module arrows, or legend until the figure reads well.

## Step 1: Write a spec file

Create a Python file (anywhere on disk) that exports a top-level `SPEC` variable. The spec file may import `Block` directly — the renderer adds its own directory to `sys.path` before executing the spec.

```python
# my_model_spec.py
from render_model import Block

SPEC = Block("MyTransformer", kind="model", children=[
    Block("Inputs\ntokens · positions · masks", kind="io"),
    Block("Encoder × N", kind="encoder_layer", children=[
        Block("Multi-Head Attention\nQ · K · V", kind="attn"),
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

### Recognised `kind` values (built-in palette)

- `io` — inputs, outputs (light blue)
- `embedding`, `emb` — embeddings (purple)
- `proj`, `head`, `decoder` — projections / output heads (peach)
- `attn`, `feat_attn` — attention (red / pink)
- `sample_attn` — sample/set attention (blue)
- `mlp` — feed-forward / MLP (green)
- `encoder`, `encoder_layer` — encoder containers (orange; only visible if rendered as a leaf at a coarse depth)
- `model` — root container (white)

Add your own kinds via the optional `PALETTE` export (see below).

## Matrix-shaped blocks (heatmap inputs/outputs)

For data-tensor visualisation, use the `Matrix` constructor instead of a plain `Block`. It renders the leaf as a small heatmap grid instead of a coloured box — perfect for showing what a model consumes and produces.

### Flat layout (single row group)

```python
from render_model import Block, Matrix

Matrix("Inputs\nN samples × features", kind="io", rows=5, groups=[
    ("modality A", 10, 0.00, "#9CB3E5"),
    ("modality B",  8, 0.50, "#88B04B"),
])
```

Each `groups` entry is a tuple of `(label, ncols, missing_frac, color)`:
- `label` — text shown below the matrix for that column group; pass `""` to hide.
- `ncols` — number of columns this group occupies.
- `missing_frac` — fraction of cells (in this group) rendered in light grey to indicate missingness. `0.0` = fully observed.
- `color` — base colour for cells in this group; pass `None` to fall back to the block's `kind` colour. Per-group colours are useful for distinguishing modalities (e.g. modality A blue, modality B green, predictions red).

### Row groups (rows with different roles)

When rows have different roles — e.g. support examples vs query rows with different missingness — pass `row_groups` instead. Each row group has its own label, row count, and column-group list:

```python
Matrix("Inputs\nN samples × (A + B)", kind="io", row_groups=[
    ("support", 3, [
        ("modality A",                     10, 0.00, "#9CB3E5"),
        ("modality B with missing values",  8, 0.20, "#88B04B"),
    ]),
    ("query", 2, [
        ("modality A",                     10, 0.00, "#9CB3E5"),
        ("modality B with missing values",  8, 0.80, "#88B04B"),
    ]),
])
```

Row-group labels are rendered along the left edge of the matrix (one label per group, vertically centred over its rows). Column labels still come from the first row group's `column_groups` list. All row groups within a matrix should share the same column structure (column counts and labels); only `missing_frac` and `color` should vary per group.

The rendering is **deterministic** (seeded with `seed`, default 42), so the same spec always produces the same pattern of cell shades and missing cells.

### Transposing the matrix

Pass `transposed=True` to flip rows ↔ columns at render time:

```python
Matrix("Inputs\nN samples × (A + B)", kind="io",
       transposed=True, row_groups=[...])
```

The spec stays in the natural "samples × features" orientation; the renderer transposes both the cell grid and the labels (row-group labels move to the bottom, column-group labels move to the left) so the **features axis becomes vertical**. Use this when:

- The feature axis is much longer than the sample axis, and a wide matrix is squashing the figure horizontally.
- The bottom column-label text is wider than the matrix itself, stretching the cells; transposing moves the long labels to the left, where they no longer constrain the matrix's horizontal extent.

For a balanced look, set `transposed=True` consistently on all I/O matrices in the same diagram so they share the same orientation.

### Title and label placement for `Matrix` blocks

A matrix block is rendered as three sibling sub-tables inside the coloured block:

- **Title** (e.g. `Inputs`) — *outside* the coloured box, above it, like every other block's title.
- **Subtitle** (e.g. `N samples × (A + B)`) — *inside* the coloured box, at the top.
- **Row-group labels** (e.g. `modality A`, `modality B with missing values`) — *outside* the bordered heatmap grid, stacked to the left of it; each label's vertical centre is aligned with its row group. Long phrases wrap to two balanced lines automatically.
- **Column-group labels** (e.g. `support`, `query`) — *outside* the bordered heatmap grid, in a row below it; each label's `WIDTH` is fixed to its column group's pixel span, with balanced wrap targeted at that width.

Keeping the row/column labels in *sibling* sub-tables (instead of `ROWSPAN` / `COLSPAN` cells inside the grid's `TABLE`) is what guarantees the heatmap cells stay `FIXEDSIZE` 9×9 pixels regardless of how long the labels are — a wide label like `modality B with missing values` can no longer stretch the cells horizontally or vertically.

### When to use a `Matrix` block

- Showing what a tensor input/output looks like (e.g. samples × features, sparse vs dense).
- Distinguishing rows by role (support / query, train / test, masked / unmasked) using row groups.
- Comparing observed vs imputed/predicted modalities side by side.
- Any block whose meaning is best conveyed as "this is a table of values".

`Matrix` blocks may be mixed freely with regular `Block`s in the same spec — typically use them only at the I/O boundary of the diagram.

## Step 2: Render

```bash
# Top-level summary
python scripts/render_model.py --spec my_model_spec.py \
    --depth 1 --out plots/my_model_d1.png

# Direct children of the root
python scripts/render_model.py --spec my_model_spec.py \
    --depth 2 --out plots/my_model_d2.png

# Editable vector output for Illustrator / Inkscape
python scripts/render_model.py --spec my_model_spec.py \
    --depth 3 --out plots/my_model_d3.svg
```

Depth peels one layer of abstraction at a time:

| `--depth` | Shows |
|---|---|
| `1` | One big root block. |
| `2` | Direct children of the root. |
| `3` | Grandchildren. |
| `4+` | Deeper. Stop when it stops being useful — usually ≤ 4. |

## Step 3: Customisation

| Goal | How |
|---|---|
| Editable vector output | Use `.svg` in `--out`. The SVG has one `<g>` per box with `<linearGradient>` fills and `<text>` labels. |
| Colour scheme | `--scheme {default,dark,colorful,minimal}` |
| Vertical (top→bottom) layout | `--rankdir TB` (default is `LR`) |
| Suppress intra-module `↓` arrows | `--no_intra_arrows` |
| Suppress the colour legend | `--no_legend` |
| Tree-print to terminal | `--mode text` |
| Add custom colour kinds | In the spec file, export `PALETTE = {"my_kind": "#hex"}`. Merged on top of the chosen scheme. |
| Custom legend entries | In the spec file, export `LEGEND_ENTRIES = [("kind", "Display Label"), ...]`. Replaces the default six. |

## Design conventions

When defining a spec, follow these conventions so the figure reads well:

- **Short titles + subtitles.** Put a brief name first, then `\n`, then a short description. The renderer word-wraps both automatically (titles ≤ 30 chars/line, descriptions ≤ 40 chars/line, balanced so two-line wraps split near the middle).
- **Consistent `kind` semantics.** Two blocks with the same kind should mean the same thing — that's what the legend promises the reader.
- **Use `kind="io"` for top-level inputs/outputs.** This guarantees the conventional light-blue I/O boxes at the ends of the flow.
- **Mark virtual wrappers with `flatten_when_expanded=True`.** Avoids a redundant outer box once you expand into its children.
- **Stay shallow.** Aim for depth ≤ 4. Deeper trees usually mean you should consolidate at a higher abstraction level.
- **Hand-curate, don't auto-introspect.** PyTorch module trees are too noisy and parameter-heavy to render directly; write the spec at the level of conceptual blocks (Attention, MLP, Embedding) rather than the level of `nn.Linear`, `nn.Dropout`, `nn.LayerNorm`.

## What the renderer guarantees

- **Title above each box** (larger, bold) using a fixed per-level font cascade (18 → 14 → 12 → 11 pt).
- **Subtitle inside the box** (smaller) at a uniform 10 pt across the whole figure, regardless of nesting depth — so identical-level boxes look identical across modules.
- **Containers are neutral** (white in light schemes, dark in `--scheme dark`); only leaf boxes carry semantic colour.
- **Within-module flow arrows** (`↓`) between consecutive children, plus cross-module arrows attached to the midpoint of each box's border.
- **A legend** at the bottom mapping every leaf colour to its meaning.

## Reference examples

The repository ships with four complete example specs the agent can copy and adapt:

- `examples/transformer_spec.py` — a minimal Transformer encoder, `Block`-only. Good starting point for a stack-of-layers architecture.
- `examples/matrix_io_spec.py` — a set-attention imputer with `Matrix` I/O, support/query row groups, missingness masks, and a nested encoder/decoder backbone. Good starting point for any model with structured tensor inputs/outputs.
- `examples/gears_spec.py` — GEARS, a dual-GNN perturbation response model. Good starting point for graph-based architectures and shows how to register a custom `kind` (`gnn`) via the `PALETTE` export.
- `examples/biolord_spec.py` — biolord, a disentangled attribute-aware autoencoder with parallel latent branches that merge into a concatenation node. Good starting point for showing parallel encoder branches via `flow=False`.

Pre-rendered depth-3 SVGs for the last two live in `examples/figures/` (`gears_d3.svg`, `biolord_d3.svg`); the README embeds them as a gallery.

Render any spec at any depth with:

```bash
python scripts/render_model.py --spec examples/transformer_spec.py \
    --depth 3 --out plots/transformer_d3.svg

python scripts/render_model.py --spec examples/matrix_io_spec.py \
    --depth 3 --out plots/matrix_io_d3.svg

python scripts/render_model.py --spec examples/gears_spec.py \
    --depth 3 --out plots/gears_d3.svg

python scripts/render_model.py --spec examples/biolord_spec.py \
    --depth 3 --out plots/biolord_d3.svg
```

## Anti-patterns

- ❌ **Auto-introspecting a `nn.Module`.** The resulting tree is too granular (one block per `nn.Linear`); the figure becomes unreadable. Hand-curate the spec at the level of conceptual blocks.
- ❌ **Long single-line `name` strings.** They make boxes wide. Always put a `\n` between the short title and its description.
- ❌ **Reusing one `kind` for unrelated blocks.** The legend will mislead the reader. Add a new kind via `PALETTE` and `LEGEND_ENTRIES`.
- ❌ **Depth > 5.** Almost always means the spec is over-decomposed. Roll lower-level blocks back into their parents.

## Files in this skill

| Path | Purpose |
|---|---|
| `scripts/render_model.py` | Generic renderer + CLI. Exports `Block` and `Matrix` for spec files to import. |
| `examples/transformer_spec.py` | Reference spec: minimal Transformer encoder. Copy and edit when defining your own. |
| `examples/matrix_io_spec.py` | Reference spec: matrix-shape Inputs / Outputs with support/query row groups and a nested encoder/decoder. |
| `examples/gears_spec.py` | Reference spec: GEARS dual-GNN perturbation response model; shows `PALETTE` export for a custom `gnn` kind. |
| `examples/biolord_spec.py` | Reference spec: biolord disentangled autoencoder; shows parallel encoder branches via `flow=False`. |
| `examples/figures/` | Pre-rendered depth-3 SVGs (`gears_d3.svg`, `biolord_d3.svg`) embedded in the README gallery. |
| `README.md` | Library-style usage and CLI reference (also useful when invoking the tool by hand). |

## Recommended agent workflow

When asked to draw a model:

1. **Ask only what you must.** If the user has just described the architecture in text or has a paper/code at hand, write the spec yourself rather than asking dozens of clarifying questions. Pick reasonable `kind`s from the built-in palette.
2. **Start at `--depth 2`.** It usually gives a one-glance summary of the model. Render `--depth 3` next to expose the most informative sub-modules. Only go deeper (4+) if the user explicitly asks.
3. **Prefer SVG** for any output the user will edit, paste into a slide, or include in a paper.
4. **Iterate the spec, not the renderer.** If the figure looks off, the fix is almost always in the spec (titles, kinds, `flatten_when_expanded`, `Matrix` row groups), not in the renderer code.
5. **Keep specs in version control next to the model code.** A spec is documentation: re-rendering it after a model change should produce the new figure for free.
