"""Generic architecture-diagram renderer.

This script turns a hand-curated *spec file* describing a neural-network
architecture into a publication-ready Graphviz diagram. The spec file is
a plain Python module that exports:

    SPEC             — a ``Block`` tree describing the architecture
                       (required).
    LEGEND_ENTRIES   — list of ``(kind, "Human Label")`` pairs to render
                       as the bottom legend (optional; defaults to the
                       six built-in semantic kinds).
    PALETTE          — extra/overriding ``kind → "#hexcolor"`` mappings
                       (optional; merged on top of the chosen scheme).

The spec file may ``from render_model import Block`` — this script adds
its own directory to ``sys.path`` before importing the spec, so no path
gymnastics are needed.

Usage
-----
    python render_model.py --spec my_model_spec.py \
        --depth 3 --out plots/my_model_d3.svg

Output format is inferred from the ``--out`` suffix: ``.png``, ``.svg``,
``.pdf``, or ``.dot``. The emitted SVG uses one ``<g>`` per box with
``<linearGradient>`` fills and ``<text>`` labels — it opens cleanly in
Adobe Illustrator, Inkscape, or Photoshop for further editing.

Conceptual depth peels one layer of abstraction at a time:
    depth=1  one big root block
    depth=2  direct children of the root
    depth=3  grandchildren
    depth=4+ deeper

See ``examples/transformer_spec.py`` for a complete example.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Block data class (this is what spec files import)
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One labelled box (or cluster of boxes) in the architecture diagram.

    Fields
    ------
    name : str
        The display name. Everything before the first ``\\n`` becomes the
        bold title rendered *above* the coloured box; everything after
        becomes the smaller subtitle rendered *inside* the box.

    kind : str
        Semantic category used to look up the leaf-box fill colour in
        the palette. Container blocks (those with ``children``) ignore
        their kind because container fill is always neutral.

    children : list[Block]
        Sub-blocks. When the rendering depth makes them visible, they
        are stacked vertically inside this block. When they're not
        visible, this block is rendered as a leaf using its description.

    flow : bool, default True
        When True (default), a small ↓ arrow is drawn between
        consecutive children to indicate dataflow. Set False for groups
        of parallel siblings where ordering doesn't apply.

    flatten_when_expanded : bool, default False
        When True, the wrapping block disappears once its children
        become visible — the children take its slot in the parent's
        chain. Useful for "virtual" wrappers that exist only to group
        sub-blocks at coarse depth (e.g. a single "Backbone" box at
        depth 1 that should not exist at depth 2+).

    matrix_rows / matrix_groups / matrix_row_groups / matrix_seed
        Optional *heatmap-matrix* rendering. When ``matrix_groups`` or
        ``matrix_row_groups`` is non-empty the block renders as a small
        heatmap instead of a coloured leaf box — useful for visualising
        data tensors (e.g. samples × (modality A + modality B) inputs
        with a missing-value mask, or a fully imputed output matrix).

        - ``matrix_groups`` is a flat list of column groups, each a
          tuple ``(label, ncols, missing_frac, colour)``. ``colour`` can
          be ``None`` to fall back to the block's kind colour. Use this
          when every row uses the same observation pattern.
        - ``matrix_row_groups`` is a list of row groups, each a tuple
          ``(row_label, n_rows, [column_groups])``. Use this when rows
          have different roles (e.g. support vs query) with different
          missingness patterns. All row groups within a matrix should
          have the same column structure (column counts and labels);
          only ``missing_frac`` and ``colour`` should vary per row group.
        - ``matrix_seed`` controls the pseudo-random cell shades and
          missingness pattern; rendering is fully deterministic.
    """

    name: str
    kind: str = "default"
    children: list["Block"] = field(default_factory=list)
    flow: bool = True
    flatten_when_expanded: bool = False
    matrix_rows: int = 5
    matrix_groups: list[tuple[str, int, float, str | None]] = field(
        default_factory=list)
    matrix_row_groups: list[
        tuple[str, int, list[tuple[str, int, float, str | None]]]
    ] = field(default_factory=list)
    matrix_seed: int = 42
    matrix_transposed: bool = False


def Matrix(name: str, kind: str = "io", *,
           rows: int = 5,
           groups: list[tuple[str, int, float, str | None]] | None = None,
           row_groups: list[
               tuple[str, int, list[tuple[str, int, float, str | None]]]
           ] | None = None,
           seed: int = 42,
           transposed: bool = False) -> Block:
    """Convenience constructor for a matrix-shaped Block.

    Two modes:

    - **Single row group** (default): pass ``rows`` and ``groups``.
      ``groups`` is a list of ``(label, ncols, missing_frac, colour)``.
    - **Multiple row groups**: pass ``row_groups``, a list of
      ``(row_label, n_rows, column_groups)`` tuples. Use this when rows
      have different roles (e.g. support vs query) with different
      missingness patterns.

    Example (single row group)::

        Matrix("Inputs\\nsamples x features", kind="io", rows=5, groups=[
            ("modality A", 10, 0.0, "#9CB3E5"),
            ("modality B",  8, 0.5, "#88B04B"),
        ])

    Example (row groups)::

        Matrix("Inputs", kind="io", row_groups=[
            ("support", 3, [
                ("modality A", 10, 0.0, "#9CB3E5"),
                ("modality B",  8, 0.2, "#88B04B"),
            ]),
            ("query", 2, [
                ("modality A", 10, 0.0, "#9CB3E5"),
                ("modality B",  8, 0.8, "#88B04B"),
            ]),
        ])
    """
    if row_groups:
        return Block(name=name, kind=kind,
                     matrix_row_groups=list(row_groups),
                     matrix_seed=seed,
                     matrix_transposed=transposed)
    if groups is None:
        groups = [("", 10, 0.0, None)]
    return Block(name=name, kind=kind, matrix_rows=rows,
                 matrix_groups=groups, matrix_seed=seed,
                 matrix_transposed=transposed)


# ---------------------------------------------------------------------------
# Depth resolution
# ---------------------------------------------------------------------------


def _shallow(block: Block) -> Block:
    """Copy a block's leaf-rendering attributes without its children.

    Used by ``_resolve`` when collapsing a block into a leaf at the
    current rendering depth. Preserves the matrix-shape fields so a
    matrix-rendered block continues to render as a matrix when shown as
    a leaf at coarse depths.
    """
    return Block(
        name=block.name,
        kind=block.kind,
        flow=block.flow,
        matrix_rows=block.matrix_rows,
        matrix_groups=list(block.matrix_groups),
        matrix_row_groups=list(block.matrix_row_groups),
        matrix_seed=block.matrix_seed,
        matrix_transposed=block.matrix_transposed,
    )


def _resolve(block: Block, depth: int, level: int = 0) -> list[Block]:
    """Produce the depth-trimmed render tree as a flat list of siblings.

    A block whose children are not yet visible (``level + 1 >= depth``)
    returns itself as a single leaf. An expanded block returns either
    a single new Block wrapping the resolved children (default), or the
    resolved children themselves if ``flatten_when_expanded`` is set.
    """
    if not block.children or level + 1 >= depth:
        return [_shallow(block)]
    resolved: list[Block] = []
    for c in block.children:
        resolved.extend(_resolve(c, depth, level + 1))
    if block.flatten_when_expanded:
        return resolved
    wrapper = _shallow(block)
    wrapper.children = resolved
    return [wrapper]


def _resolve_top(spec: Block, depth: int) -> list[Block]:
    """Top-level resolution: the spec root is always transparent — its
    direct children form the main chain. ``depth`` then governs whether
    each child stays a leaf (d=1), expands one level (d=2), etc."""
    resolved: list[Block] = []
    for c in spec.children:
        resolved.extend(_resolve(c, depth, level=0))
    return resolved


# ---------------------------------------------------------------------------
# Colour schemes
# ---------------------------------------------------------------------------


_PALETTE_DEFAULT = {
    "model":         "#FFFFFF",
    "io":            "#D6E4FF",
    "embedding":     "#E0BBE4",
    "encoder":       "#FFD6A5",
    "encoder_layer": "#FFD6A5",
    "feat_attn":     "#FFB6B9",
    "sample_attn":   "#A7D3F2",
    "attn":          "#FFB6B9",
    "mlp":           "#C7E9B0",
    "decoder":       "#FFE5B4",
    "head":          "#FFE5B4",
    "proj":          "#FFE5B4",
    "emb":           "#E0BBE4",
}

_PALETTE_DARK = {
    "model":         "#222222",
    "io":            "#1F3A60",
    "embedding":     "#542B68",
    "encoder":       "#7A4A1A",
    "encoder_layer": "#7A4A1A",
    "feat_attn":     "#8C1F3C",
    "sample_attn":   "#1F4E8C",
    "attn":          "#8C1F3C",
    "mlp":           "#2D5E3D",
    "decoder":       "#705F1F",
    "head":          "#705F1F",
    "proj":          "#705F1F",
    "emb":           "#542B68",
}

_PALETTE_COLORFUL = {
    "model":         "#FFFFFF",
    "io":            "#92A8D1",
    "embedding":     "#6B5B95",
    "encoder":       "#F5DF4D",
    "encoder_layer": "#F5DF4D",
    "feat_attn":     "#DD4124",
    "sample_attn":   "#009B77",
    "attn":          "#DD4124",
    "mlp":           "#88B04B",
    "decoder":       "#FF6F61",
    "head":          "#FF6F61",
    "proj":          "#FF6F61",
    "emb":           "#6B5B95",
}

# A scheme has:
#   palette          : kind → fill colour for *leaf* boxes (carries meaning)
#   bg               : graph background
#   font             : default text colour
#   container_bg     : fill for container boxes (no semantic meaning)
#   container_border : thin outline for container boxes
_SCHEMES = {
    "default":  {"palette": _PALETTE_DEFAULT,  "bg": "white",   "font": "#222222",
                 "container_bg": "#FFFFFF", "container_border": "#BBBBBB"},
    "dark":     {"palette": _PALETTE_DARK,     "bg": "#1c1c1c", "font": "#EEEEEE",
                 "container_bg": "#1c1c1c", "container_border": "#666666"},
    "colorful": {"palette": _PALETTE_COLORFUL, "bg": "white",   "font": "#222222",
                 "container_bg": "#FFFFFF", "container_border": "#BBBBBB"},
    "minimal":  {"palette": {},                "bg": "white",   "font": "#222222",
                 "container_bg": "#FFFFFF", "container_border": "#BBBBBB"},
}


def _color_for(block: Block, scheme: dict) -> str:
    palette = scheme["palette"]
    if not palette:
        return "#FFFFFF" if scheme["bg"] == "white" else "#333333"
    return palette.get(block.kind, "#F0F0F0")


def _shift_color(hex_color: str, t: float) -> str:
    """Linearly shift a ``#rrggbb`` colour: ``t > 0`` lightens toward white,
    ``t < 0`` darkens toward black. ``t`` is clamped to ``[-1, 1]``.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        r += int((255 - r) * t)
        g += int((255 - g) * t)
        b += int((255 - b) * t)
    else:
        r = int(r * (1.0 + t))
        g = int(g * (1.0 + t))
        b = int(b * (1.0 + t))
    return (f"#{max(0, min(255, r)):02x}"
            f"{max(0, min(255, g)):02x}"
            f"{max(0, min(255, b)):02x}")


# ---------------------------------------------------------------------------
# Graphviz rendering (HTML-table labels)
# ---------------------------------------------------------------------------


_TITLE_SIZES = [18, 14, 12, 11]
_BODY_SIZE = 10
# Wrap thresholds are intentionally generous so that short labels (e.g.
# "Bar-Distribution Head" at 21 chars, or "predicted protein expression"
# at 27 chars) stay on a single line. Only genuinely long phrases wrap.
_TITLE_WRAP_CHARS = 30
_BODY_WRAP_CHARS = 40


def _title_size_for(level: int) -> int:
    if level >= len(_TITLE_SIZES):
        return _TITLE_SIZES[-1]
    return _TITLE_SIZES[level]


def _wrap_text(text: str, max_chars: int) -> str:
    """Balanced word-wrap.

    Wraps each paragraph so that:
      1. The number of lines is the minimum achievable under ``max_chars``
         (matches greedy wrap's line count).
      2. The longest line in the wrap is as short as possible, so two-line
         wraps split near the middle instead of leaving the second line
         tiny. E.g. ``cross-attention over the support set context`` (43
         ch, max 40) wraps to ``cross-attention over the`` / ``support
         set context`` instead of the greedy ``cross-attention over the
         support`` / ``set context``.
    """
    out: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            out.append("")
            continue
        total = sum(len(w) for w in words) + len(words) - 1
        if total <= max_chars:
            out.append(" ".join(words))
            continue
        word_lens = [len(w) for w in words]

        def fits(target: int, max_lines: int | None = None) -> bool:
            lines = 1
            cur = word_lens[0]
            for wl in word_lens[1:]:
                if cur + 1 + wl <= target:
                    cur += 1 + wl
                else:
                    lines += 1
                    if max_lines is not None and lines > max_lines:
                        return False
                    cur = wl
            return True

        # Greedy upper-bound on number of lines under the user's max_chars.
        n_lines = 1
        line_len = word_lens[0]
        for wl in word_lens[1:]:
            if line_len + 1 + wl <= max_chars:
                line_len += 1 + wl
            else:
                n_lines += 1
                line_len = wl

        # Binary-search the smallest target that still fits in n_lines.
        lo, hi = max(word_lens), max_chars
        while lo < hi:
            mid = (lo + hi) // 2
            if fits(mid, max_lines=n_lines):
                hi = mid
            else:
                lo = mid + 1
        target = lo

        # Re-emit greedily under that target — yields balanced lines.
        lines: list[str] = []
        cur_words = [words[0]]
        cur_len = word_lens[0]
        for w, wl in zip(words[1:], word_lens[1:]):
            if cur_len + 1 + wl <= target:
                cur_words.append(w)
                cur_len += 1 + wl
            else:
                lines.append(" ".join(cur_words))
                cur_words = [w]
                cur_len = wl
        lines.append(" ".join(cur_words))
        out.extend(lines)
    return "\n".join(out)


def _label_html(name: str) -> str:
    return (name
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<BR/>"))


def _flow_arrow_row(scheme: dict) -> str:
    return (
        '<TR><TD BORDER="0" CELLPADDING="2">'
        f'<FONT FACE="DejaVu Sans Mono" COLOR="{scheme["font"]}" '
        'POINT-SIZE="16"><B>&#8595;</B></FONT>'
        '</TD></TR>'
    )


def _box_table(content_rows: str, base: str, border: str,
               light: str, *, padding: str = "6", spacing: str = "4") -> str:
    return (
        f'<TABLE BORDER="2" COLOR="{border}" CELLBORDER="0" '
        f'CELLPADDING="{padding}" CELLSPACING="{spacing}" '
        f'BGCOLOR="{light}:{base}" GRADIENTANGLE="270" STYLE="ROUNDED">'
        f'{content_rows}'
        '</TABLE>'
    )


def _cell_intensity(r: int, c: int, gi: int, seed: int, channel: str) -> float:
    """Deterministic pseudo-random value in [0, 1) for one heatmap cell.

    ``channel`` is a stable string identifier ("value" vs "missing-mask")
    so the same cell's "is missing" and "value" rolls don't correlate.
    Uses a Knuth-style hash so behaviour is identical across Python
    versions (which is *not* true of the built-in ``hash`` for strings).
    """
    h = (r * 73856093) ^ (c * 19349663) ^ (gi * 83492791) ^ (seed * 2971215073)
    if channel == "missing":
        h ^= 0x9E3779B9
    h &= 0xFFFFFFFF
    h = (h * 2654435761) & 0xFFFFFFFF
    return h / 0x100000000


def _build_matrix_grid(block: Block, scheme: dict, missing_color: str
                       ) -> tuple[list[list[str]],
                                  list[tuple[str, int]],
                                  list[tuple[str, int]]]:
    """Compute a normalized cell-colour grid + label specs for the matrix.

    Returns
    -------
    cells : list[list[str]]
        2D ``[n_rows][n_cols]`` of ``"#rrggbb"`` colour strings.
    row_label_groups : list[(label, count)]
        Left-side row-group labels.
    col_label_groups : list[(label, count)]
        Bottom-side column-group labels.

    Honours ``block.matrix_transposed``: when set, the grid is
    transposed and the row/col label specs are swapped so the renderer
    only needs to handle a single orientation.
    """
    seed = block.matrix_seed
    if block.matrix_row_groups:
        row_groups = block.matrix_row_groups
    else:
        col_groups = block.matrix_groups or [("", 10, 0.0, None)]
        row_groups = [("", block.matrix_rows, col_groups)]

    first_col_groups = row_groups[0][2]
    n_rows = sum(rg[1] for rg in row_groups)
    n_cols = sum(cg[1] for cg in first_col_groups)

    cells: list[list[str]] = [[missing_color] * n_cols for _ in range(n_rows)]
    global_r = 0
    for rg_idx, (_, n_rg_rows, rg_col_groups) in enumerate(row_groups):
        for _r in range(n_rg_rows):
            global_c = 0
            for gi, (_, ncols, missing_frac, color) in enumerate(rg_col_groups):
                base = color or _color_for(block, scheme)
                for _c in range(ncols):
                    m = _cell_intensity(global_r, global_c, gi, seed, "missing")
                    if m < missing_frac:
                        cells[global_r][global_c] = missing_color
                    else:
                        v = _cell_intensity(global_r, global_c, gi, seed, "value")
                        cells[global_r][global_c] = _shift_color(
                            base, 0.35 - v * 0.9)
                    global_c += 1
            global_r += 1

    row_label_groups = [(lbl, count) for lbl, count, _ in row_groups]
    col_label_groups = [(lbl, count) for lbl, count, _, _ in first_col_groups]

    if block.matrix_transposed:
        cells = [[cells[r][c] for r in range(n_rows)] for c in range(n_cols)]
        row_label_groups, col_label_groups = col_label_groups, row_label_groups

    return cells, row_label_groups, col_label_groups


def _matrix_grid_only_html(cells: list[list[str]], cell_size: int) -> str:
    """Render just the colored heatmap cells as a bordered TABLE.

    Cells are ``FIXEDSIZE=TRUE`` so the grid keeps its natural ``n_cols
    × cell_size`` width and ``n_rows × cell_size`` height no matter what
    surrounds it — in particular, no row- or column-label text can
    stretch the cells.
    """
    if not cells or not cells[0]:
        return ""
    body_rows: list[str] = []
    for row in cells:
        tds = "".join(
            f'<TD WIDTH="{cell_size}" HEIGHT="{cell_size}" '
            f'FIXEDSIZE="TRUE" BGCOLOR="{c}"></TD>'
            for c in row
        )
        body_rows.append(f'<TR>{tds}</TR>')
    return (
        '<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" '
        'COLOR="#888888">'
        + "".join(body_rows) +
        '</TABLE>'
    )


# Approx horizontal pixels-per-character for the (smaller) col-label font.
# Used to pick the wrap target for a column-group label given its
# group's pixel width. Slightly conservative on purpose so labels fit.
_MATRIX_COL_LABEL_CHAR_PX = 5
# Soft target for wrapping a row-group label across multiple lines so a
# long phrase like "modality B with missing values" naturally breaks
# into two roughly balanced lines.
_MATRIX_ROW_LABEL_WRAP = 16


def _row_labels_outside_html(row_label_groups: list[tuple[str, int]],
                             cell_size: int, scheme: dict) -> str:
    """Render row-group labels stacked vertically to the left of the
    matrix grid. Each label cell's ``HEIGHT`` is set to its group's
    pixel height (``count × cell_size``), so the label vertically
    centers over its group of rows. Text is word-wrapped (balanced) so
    long phrases naturally break into two lines.
    """
    if not any(lbl for lbl, _ in row_label_groups):
        return ""
    rows: list[str] = []
    for label, count in row_label_groups:
        height = count * cell_size
        if label:
            wrapped = _wrap_text(label, _MATRIX_ROW_LABEL_WRAP)
            text = (
                f'<FONT POINT-SIZE="{_BODY_SIZE}" '
                f'COLOR="{scheme["font"]}">'
                f'{_label_html(wrapped)}</FONT>'
            )
            rows.append(
                f'<TR><TD HEIGHT="{height}" ALIGN="RIGHT" '
                f'VALIGN="MIDDLE" CELLPADDING="4">{text}</TD></TR>'
            )
        else:
            rows.append(f'<TR><TD HEIGHT="{height}"></TD></TR>')
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        + "".join(rows) +
        '</TABLE>'
    )


def _col_labels_outside_html(col_label_groups: list[tuple[str, int]],
                             cell_size: int, scheme: dict) -> str:
    """Render column-group labels in a single horizontal row below the
    matrix grid. Each label cell's ``WIDTH`` is set to its group's
    pixel width; text is balance-wrapped so phrases that don't fit on
    one line break into two near-equal lines. The grid itself sits in a
    sibling TABLE row, so even if a label needs to overflow its
    column's pixel width, no matrix cell gets stretched.
    """
    if not any(lbl for lbl, _ in col_label_groups):
        return ""
    cells: list[str] = []
    for label, count in col_label_groups:
        width = count * cell_size
        if label:
            wrap_chars = max(4, width // _MATRIX_COL_LABEL_CHAR_PX)
            wrapped = _wrap_text(label, wrap_chars)
            text = (
                f'<FONT POINT-SIZE="{max(_BODY_SIZE - 1, 8)}" '
                f'COLOR="{scheme["font"]}">'
                f'{_label_html(wrapped)}</FONT>'
            )
            cells.append(
                f'<TD WIDTH="{width}" ALIGN="CENTER" '
                f'CELLPADDING="2">{text}</TD>'
            )
        else:
            cells.append(f'<TD WIDTH="{width}"></TD>')
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR>{"".join(cells)}</TR>'
        '</TABLE>'
    )


def _matrix_html(block: Block, scheme: dict, *,
                 missing_color: str = "#E5E5E5",
                 cell_size: int = 9) -> str:
    """Render a block as a small heatmap-style grid with labels around it.

    Layout (2×2 outer TABLE, all sibling sub-tables independent):

    .. code-block:: text

       +-------------+----------------+
       | row labels  | matrix grid    |
       +-------------+----------------+
       |             | col labels     |
       +-------------+----------------+

    Why three tables instead of one: keeping row- and column-group
    labels in *sibling* sub-tables (rather than inside the bordered
    grid's TABLE) means the grid's ``FIXEDSIZE`` cells can't be
    stretched horizontally by a wide ``COLSPAN`` label nor vertically
    by a tall ``ROWSPAN`` label. Long labels word-wrap into two lines
    via the balanced wrapper; very narrow column groups whose label
    can't fit on one line still wrap, and the (rare) leftover overflow
    is purely cosmetic — the matrix grid stays compact.
    """
    cells, row_label_groups, col_label_groups = _build_matrix_grid(
        block, scheme, missing_color)

    grid_html = _matrix_grid_only_html(cells, cell_size)
    row_labels_html = _row_labels_outside_html(
        row_label_groups, cell_size, scheme)
    col_labels_html = _col_labels_outside_html(
        col_label_groups, cell_size, scheme)

    has_row_labels = bool(row_labels_html)
    has_col_labels = bool(col_labels_html)

    parts = [
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
    ]
    if has_row_labels:
        parts.append(
            '<TR>'
            f'<TD VALIGN="TOP">{row_labels_html}</TD>'
            f'<TD ALIGN="LEFT">{grid_html}</TD>'
            '</TR>'
        )
        if has_col_labels:
            parts.append(
                '<TR>'
                '<TD></TD>'
                f'<TD ALIGN="LEFT">{col_labels_html}</TD>'
                '</TR>'
            )
    else:
        parts.append(f'<TR><TD ALIGN="LEFT">{grid_html}</TD></TR>')
        if has_col_labels:
            parts.append(
                f'<TR><TD ALIGN="LEFT">{col_labels_html}</TD></TR>'
            )
    parts.append('</TABLE>')
    return "".join(parts)


def _block_html(block: Block, scheme: dict, *, intra_arrows: bool = True,
                level: int = 0) -> str:
    """Render a block as an outer transparent table:

        Row 1 — bold title above the coloured box (size = title_size_for(level)).
        Row 2 — (matrix blocks only) the body description, also outside
                the box so wide subtitles don't pad the matrix.
        Row 3 — the coloured rounded box: either the in-box description
                (non-matrix leaves) or the matrix grid (matrix leaves)
                or recursively rendered children for containers.

    Container fill is neutral; only leaves carry semantic colour. Long
    titles and descriptions are balanced-wrapped so boxes stay compact.
    """
    title_text, _, body_text = block.name.partition("\n")
    title_html = _label_html(_wrap_text(title_text, _TITLE_WRAP_CHARS))
    body_html = _label_html(_wrap_text(body_text, _BODY_WRAP_CHARS))
    title_size = _title_size_for(level)

    if not block.children:
        leaf_base = _color_for(block, scheme)
        leaf_light = _shift_color(leaf_base, 0.30)
        leaf_border = _shift_color(leaf_base, -0.20)
        if block.matrix_groups or block.matrix_row_groups:
            matrix_html = _matrix_html(block, scheme)
            subtitle_row = (
                '<TR><TD CELLPADDING="4" ALIGN="CENTER">'
                f'<FONT POINT-SIZE="{_BODY_SIZE}">{body_html}</FONT>'
                '</TD></TR>'
                if body_text else ""
            )
            inner = (
                subtitle_row
                + f'<TR><TD CELLPADDING="6">{matrix_html}</TD></TR>'
            )
            box = _box_table(inner, leaf_base, leaf_border, leaf_light,
                             padding="6", spacing="0")
        else:
            if body_text:
                inner = (
                    f'<TR><TD CELLPADDING="8">'
                    f'<FONT POINT-SIZE="{_BODY_SIZE}">{body_html}</FONT>'
                    f'</TD></TR>'
                )
            else:
                inner = '<TR><TD CELLPADDING="14">&nbsp;</TD></TR>'
            box = _box_table(inner, leaf_base, leaf_border, leaf_light,
                             padding="6", spacing="0")
    else:
        container_bg = scheme["container_bg"]
        container_border = scheme["container_border"]
        rows: list[str] = []
        show_arrows = intra_arrows and block.flow
        for i, c in enumerate(block.children):
            if i > 0 and show_arrows:
                rows.append(_flow_arrow_row(scheme))
            rows.append(
                '<TR><TD CELLPADDING="0">'
                f'{_block_html(c, scheme, intra_arrows=intra_arrows, level=level + 1)}'
                '</TD></TR>'
            )
        box = _box_table("".join(rows), container_bg, container_border,
                         container_bg)

    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        '<TR><TD CELLPADDING="3" ALIGN="CENTER">'
        f'<FONT POINT-SIZE="{title_size}" COLOR="{scheme["font"]}">'
        f'<B>{title_html}</B></FONT>'
        '</TD></TR>'
        f'<TR><TD CELLPADDING="0">{box}</TD></TR>'
        '</TABLE>'
    )


_DEFAULT_LEGEND_ENTRIES = [
    ("io",          "Input / Output"),
    ("emb",         "Embedding"),
    ("proj",        "Projection / Head"),
    ("attn",        "Attention"),
    ("sample_attn", "Sample Attention"),
    ("mlp",         "MLP / Feed-forward"),
]


def _legend_html(scheme: dict, entries: list[tuple[str, str]]) -> str:
    cells: list[str] = []
    for kind, label in entries:
        base = scheme["palette"].get(kind, "#F0F0F0")
        light = _shift_color(base, 0.30)
        border = _shift_color(base, -0.20)
        swatch = (
            f'<TABLE BORDER="2" COLOR="{border}" CELLBORDER="0" '
            f'CELLPADDING="0" CELLSPACING="0" '
            f'BGCOLOR="{light}:{base}" GRADIENTANGLE="270" STYLE="ROUNDED">'
            '<TR><TD WIDTH="30" HEIGHT="14"></TD></TR>'
            '</TABLE>'
        )
        cells.append(
            f'<TD CELLPADDING="4">{swatch}</TD>'
            f'<TD CELLPADDING="4" ALIGN="LEFT">'
            f'<FONT POINT-SIZE="{_BODY_SIZE + 1}" COLOR="{scheme["font"]}">'
            f'{_label_html(label)}</FONT></TD>'
        )
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="6" CELLPADDING="0">'
        f'<TR>{"".join(cells)}</TR>'
        '</TABLE>'
    )


def _emit_dot(resolved_top: list[Block], scheme: dict, rankdir: str,
              *, intra_arrows: bool = True, show_legend: bool = True,
              legend_entries: list[tuple[str, str]] | None = None) -> str:
    """Render the resolved block list to Graphviz dot source.

    Containers carry no semantic colour; only leaf boxes are coloured by
    ``block.kind``. When ``show_legend`` is True, a legend is drawn as
    the graph's bottom caption mapping each leaf-box colour to its
    meaning.
    """
    tail_port, head_port = (":e", ":w") if rankdir == "LR" else (":s", ":n")
    if legend_entries is None:
        legend_entries = _DEFAULT_LEGEND_ENTRIES

    lines = [
        "digraph G {",
        f'  rankdir={rankdir};',
        f'  bgcolor="{scheme["bg"]}";',
        f'  fontcolor="{scheme["font"]}";',
        '  nodesep=0.5; ranksep=1.0;',
        f'  node [shape=plaintext, fontname="Helvetica", fontsize={_BODY_SIZE}, '
        f'fontcolor="{scheme["font"]}"];',
        f'  edge [color="{scheme["font"]}", penwidth=1.6, arrowsize=0.9];',
    ]
    if show_legend:
        lines.append('  labelloc="b"; labeljust="c";')
        lines.append(f'  label=<{_legend_html(scheme, legend_entries)}>;')

    for i, block in enumerate(resolved_top):
        html = _block_html(block, scheme, intra_arrows=intra_arrows)
        lines.append(f'  n{i} [label=<{html}>];')

    for i in range(len(resolved_top) - 1):
        lines.append(f'  n{i}{tail_port} -> n{i + 1}{head_port};')

    lines.append("}")
    return "\n".join(lines)


def _render_dot(dot_text: str, out_path: Path) -> None:
    if shutil.which("dot") is None:
        sys.exit("ERROR: 'dot' (graphviz) binary not found in PATH.")
    fmt = out_path.suffix.lstrip(".") or "png"
    if fmt not in {"png", "svg", "pdf", "dot"}:
        sys.exit(f"ERROR: unsupported output format '{fmt}'. "
                 "Use .png, .svg, .pdf, or .dot.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dot":
        out_path.write_text(dot_text)
        print(f"[OK] wrote {out_path}")
        return
    proc = subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(out_path)],
        input=dot_text, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"ERROR: dot failed: {proc.stderr}")
    print(f"[OK] wrote {out_path}")


# ---------------------------------------------------------------------------
# Text mode
# ---------------------------------------------------------------------------


def _print_tree(block: Block, prefix: str = "", is_last: bool = True) -> None:
    connector = "└─ " if is_last else "├─ "
    print(f"{prefix}{connector}{block.name.splitlines()[0]}")
    if not block.children:
        return
    next_prefix = prefix + ("   " if is_last else "│  ")
    for i, c in enumerate(block.children):
        _print_tree(c, next_prefix, i == len(block.children) - 1)


# ---------------------------------------------------------------------------
# Spec loader
# ---------------------------------------------------------------------------


def _load_spec_module(path: Path):
    """Import a user-provided spec file as a Python module.

    Adds this script's directory to ``sys.path`` first so the spec file
    can do ``from render_model import Block`` without knowing where the
    renderer lives.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    loader_spec = importlib.util.spec_from_file_location(path.stem, path)
    if loader_spec is None or loader_spec.loader is None:
        sys.exit(f"ERROR: could not load spec file: {path}")
    module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    if not hasattr(module, "SPEC"):
        sys.exit(f"ERROR: spec file {path} must export a top-level "
                 "variable named SPEC (a Block instance).")
    # Duck-typed isinstance: when this script is run as ``__main__`` the
    # ``Block`` class in ``__main__`` is a different object than the one
    # the spec imported via ``from render_model import Block``, so a real
    # isinstance() check would falsely fail.
    spec_obj = module.SPEC
    if not (hasattr(spec_obj, "name") and hasattr(spec_obj, "kind")
            and hasattr(spec_obj, "children")):
        sys.exit(f"ERROR: SPEC in {path} must be a Block instance, got "
                 f"{type(spec_obj).__name__}.")
    return module


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("--spec", required=True,
                    help="Path to a Python spec file that exports a "
                         "top-level ``SPEC`` (Block instance), and "
                         "optionally ``LEGEND_ENTRIES`` and ``PALETTE``.")
    pa.add_argument("--depth", type=int, default=2,
                    help="Conceptual depth: 1 = root only; 2 = root's "
                         "children; 3+ = deeper. Default 2.")
    pa.add_argument("--mode", default="dot", choices=["dot", "text"],
                    help="dot = graphviz diagram; text = terminal tree.")
    pa.add_argument("--scheme", default="default",
                    choices=list(_SCHEMES.keys()),
                    help="Color palette for dot mode.")
    pa.add_argument("--rankdir", default="LR", choices=["TB", "LR"],
                    help="Layout direction across modules: LR = horizontal "
                         "(default), TB = vertical stack.")
    pa.add_argument("--out", default="plots/model_architecture.png",
                    help="Output path (.png / .svg / .pdf / .dot). "
                         "Ignored in text mode.")
    pa.add_argument("--no_intra_arrows", action="store_true",
                    help="Suppress the small ↓ flow arrows drawn between "
                         "sub-blocks inside each module.")
    pa.add_argument("--no_legend", action="store_true",
                    help="Suppress the colour legend at the bottom.")
    args = pa.parse_args()

    spec_module = _load_spec_module(Path(args.spec).resolve())
    spec: Block = spec_module.SPEC
    legend_entries: list[tuple[str, str]] = getattr(
        spec_module, "LEGEND_ENTRIES", _DEFAULT_LEGEND_ENTRIES)
    extra_palette: dict[str, str] = getattr(spec_module, "PALETTE", {})

    scheme = dict(_SCHEMES[args.scheme])
    if extra_palette:
        scheme["palette"] = {**scheme["palette"], **extra_palette}

    resolved = _resolve_top(spec, args.depth)

    if args.mode == "text":
        header = f"{spec.name}  (depth {args.depth})"
        print(f"\n{header}")
        print("=" * len(header))
        for i, c in enumerate(resolved):
            _print_tree(c, is_last=(i == len(resolved) - 1))
        return

    dot_text = _emit_dot(
        resolved, scheme, args.rankdir,
        intra_arrows=not args.no_intra_arrows,
        show_legend=not args.no_legend,
        legend_entries=legend_entries,
    )
    _render_dot(dot_text, Path(args.out))


if __name__ == "__main__":
    main()
