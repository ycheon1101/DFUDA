#!/usr/bin/env python3
"""Build the README t-SNE + Cityscapes legend figure.

The default ``paper`` layout follows the Overleaf ``\\tsnelegend`` macro (see
``tools/overleaf_tsne_legend.tex``): row pitch ``\\dy``, patches ``++(0.28,0.18)``,
labels at ``\\xtext=0.45``, ``\\fontsize{8}{7}``.

Alternative: ``--style gray-panel`` (older bold + gray legend column).

Usage:
  python3 tools/render_tsne_readme_figure.py
  python3 tools/render_tsne_readme_figure.py --style gray-panel
  python3 tools/render_tsne_readme_figure.py --caption "(b) Ours"

  # Use your own legend PNG (e.g. exported from Overleaf) instead of generating it:
  python3 tools/render_tsne_readme_figure.py \\
      --legend-image resources/my_tsne_legend.png --caption ""
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import numpy as np

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

# From Overleaf \tsnelegend (tools/overleaf_tsne_legend.tex), TikZ units.
_TIKZ_DY = 0.25
_TIKZ_BOX_W = 0.28
_TIKZ_BOX_H = 0.18
_TIKZ_TEXT_X = 0.45  # \xtext — label anchor; \xbox = 0
_OVERLEAF_FONT_PT = 8  # \fontsize{8}{7}\selectfont — we scale px from row height


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    bold_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    )
    regular_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    for path in (bold_paths if bold else regular_paths):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    for path in regular_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_serif_font(size_px: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def rasterize_pdf(pdf_path: Path, dpi: int, tmp_prefix: Path) -> Path:
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            str(tmp_prefix),
        ],
        check=True,
    )
    return Path(str(tmp_prefix) + ".png")


def draw_plot_frame(
    img: Image.Image,
    width_px: int = 1,
    *,
    omit_right: bool = False,
) -> Image.Image:
    """Thin black frame around the rasterized PDF.

    With ``omit_right=True``, draw only top, left, and bottom edges so the plot
    meets an external legend without a double vertical line at the seam.
    """
    out = img.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    if not omit_right:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=width_px)
    else:
        draw.line([(0, 0), (w - 1, 0)], fill=(0, 0, 0), width=width_px)
        draw.line([(0, h - 1), (w - 1, h - 1)], fill=(0, 0, 0), width=width_px)
        draw.line([(0, 0), (0, h - 1)], fill=(0, 0, 0), width=width_px)
    return out


def compose_embedded_legend(
    plot_img: Image.Image,
    legend_frac: float,
    pad_px: int,
    bar_w: int,
    font_px: int,
    panel_rgb: Tuple[int, int, int] = (235, 235, 238),
) -> Image.Image:
    """Gray legend band + bold sans (legacy README style)."""
    w, h = plot_img.size
    text_gutter = max(200, int(font_px * 13))
    leg_w = max(int(w * legend_frac), bar_w + pad_px * 2 + text_gutter)

    out = Image.new("RGB", (w + leg_w, h), (255, 255, 255))
    out.paste(plot_img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)

    x_leg0 = w
    draw.rectangle([x_leg0, 0, w + leg_w - 1, h - 1], fill=panel_rgb)

    sep_w = 2
    draw.rectangle([x_leg0, 0, x_leg0 + sep_w - 1, h - 1], fill=(75, 75, 78))

    font = _load_font(font_px, bold=True)

    inner_pad = max(10, pad_px // 2)
    usable_h = h - 2 * inner_pad
    row_h = max(usable_h / len(CLASSES), float(font_px) + 4.0)
    sw = bar_w

    x0 = w + sep_w + pad_px
    y = float(inner_pad)

    for name, rgb in zip(CLASSES, PALETTE):
        fill = tuple(int(x) for x in rgb)
        y0, y1 = int(y), int(y + row_h) - 2
        draw.rectangle(
            [x0, y0, x0 + sw, y1],
            fill=fill,
            outline=(40, 40, 42),
            width=2,
        )
        draw.text(
            (x0 + sw + 10, int(y0 + (row_h - font_px) / 2 - 2)),
            name,
            fill=(15, 15, 18),
            font=font,
        )
        y += row_h

    return out


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    if hasattr(font, "getbbox"):
        b = font.getbbox(text)
        return b[2] - b[0]
    return font.getsize(text)[0]


def build_legend_pil_paper(
    height_px: int,
    fontsize_pt: float,
    w_frac: float,
    min_width_px: int,
) -> Image.Image:
    """Match Overleaf ``\\tsnelegend``: row pitch ``\\dy``, box ``++(0.28,0.18)``, text at ``\\xtext=0.45``."""
    inner = 10
    n = len(CLASSES)
    usable_h = max(1, height_px - 2 * inner)
    row_h = usable_h / n

    # In TikZ one row advances by \dy; box height is 0.18 → vertical scale vs row pitch.
    sw_h = row_h * (_TIKZ_BOX_H / _TIKZ_DY)
    sw_w = sw_h * (_TIKZ_BOX_W / _TIKZ_BOX_H)
    # Horizontal gap: label starts at x=0.45, box ends at 0.28 (+0.17 in TikZ x-units).
    gap = sw_w * ((_TIKZ_TEXT_X - _TIKZ_BOX_W) / _TIKZ_BOX_W)

    # ~ \fontsize{8}{7}: tie to swatch height (paper proportions); clamp for screen PNG.
    font_px = max(10, min(20, int(round(sw_h * 0.52 + 4))))
    font = _load_serif_font(font_px)
    max_label = max(_text_width(font, c) for c in CLASSES)

    leg_w = int(inner + sw_w + gap + max_label + inner + 4)
    leg_w = max(leg_w, int(height_px * w_frac * 0.95), min_width_px)

    img = Image.new("RGB", (leg_w, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Vertically center each label on the row like ``at (\xtext,-\i*\dy+0.09)`` (mid of 0.18 box).
    for i, (name, rgb) in enumerate(zip(CLASSES, PALETTE)):
        y_top = inner + i * row_h
        y_box = y_top + (row_h - sw_h) / 2.0
        y0, y1 = int(y_box), int(y_box + sw_h)
        x0, x1 = inner, int(inner + sw_w)
        fill = tuple(int(x) for x in rgb)
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(0, 0, 0), width=1)
        ty = int(y_top + (row_h - font_px) / 2)
        draw.text((int(inner + sw_w + gap + 2), ty), name, fill=(0, 0, 0), font=font)

    return img


def build_legend_mpl_paper(
    height_px: int,
    fontsize_pt: float,
    legend_dpi: int,
    w_frac: float,
    min_width_px: int,
) -> Image.Image:
    """Camera-ready style: serif text, square patches, thin black stroke, white bg."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Times New Roman",
                "Liberation Serif",
            ],
            # Paper \tsnelegend uses \fontsize{8}{7}
            "font.size": min(fontsize_pt, _OVERLEAF_FONT_PT),
            "axes.linewidth": 0,
        }
    )

    handles = [
        Patch(
            facecolor=np.asarray(rgb, dtype=float) / 255.0,
            edgecolor="black",
            linewidth=0.6,
            label=name,
        )
        for name, rgb in zip(CLASSES, PALETTE)
    ]

    h_in = height_px / legend_dpi
    min_w_in = min_width_px / float(legend_dpi)
    # Wide figure so bbox_inches='tight' is not a vertical sliver; enforce min inches.
    w_in = max(h_in * w_frac, min_w_in, 2.0)
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=legend_dpi)
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=min(fontsize_pt, _OVERLEAF_FONT_PT),
        handlelength=1.05,
        handletextpad=0.55,
        borderaxespad=0.0,
        labelspacing=0.42,
        borderpad=0.15,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=legend_dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    leg = Image.open(buf).convert("RGB")
    lw, lh = leg.size
    if lh != height_px:
        new_w = max(1, int(round(lw * (height_px / lh))))
        leg = leg.resize((new_w, height_px), _LANCZOS)
    # Prevent the "hairline" legend when mpl tight bbox is narrow.
    lw2 = leg.size[0]
    if lw2 < min_width_px:
        s = min_width_px / float(lw2)
        leg = leg.resize(
            (min_width_px, max(1, int(round(leg.size[1] * s)))),
            _LANCZOS,
        )
    if leg.size[1] != height_px:
        leg = leg.resize((leg.size[0], height_px), _LANCZOS)
    return leg


def compose_paper_paste(
    plot_img: Image.Image,
    gap_px: int,
    fontsize: float,
    leg_dpi: int,
    w_frac: float,
    min_legend_px: int,
) -> Image.Image:
    w, h = plot_img.size
    if _HAS_MPL:
        leg = build_legend_mpl_paper(h, fontsize, leg_dpi, w_frac, min_legend_px)
    else:
        leg = build_legend_pil_paper(h, fontsize, w_frac, min_legend_px)
    lw = leg.size[0]
    out = Image.new("RGB", (w + gap_px + lw, h), (255, 255, 255))
    out.paste(plot_img.convert("RGB"), (0, 0))
    out.paste(leg, (w + gap_px, 0))
    return out


def compose_external_legend_paste(
    plot_img: Image.Image,
    legend_img: Image.Image,
    gap_px: int,
) -> Image.Image:
    """Paste a user-supplied legend image to the right (height matched to plot)."""
    plot = plot_img.convert("RGB")
    leg = legend_img.convert("RGB")
    w, h = plot.size
    lw0, lh0 = leg.size
    if lh0 != h and lh0 > 0:
        new_w = max(1, int(round(lw0 * (h / float(lh0)))))
        leg = leg.resize((new_w, h), _LANCZOS)
    lw = leg.size[0]
    out = Image.new("RGB", (w + gap_px + lw, h), (255, 255, 255))
    out.paste(plot, (0, 0))
    out.paste(leg, (w + gap_px, 0))
    return out


def add_subcaption(
    img: Image.Image,
    text: str,
    font_size: int = 15,
    margin_top: int = 10,
    margin_bottom: int = 12,
) -> Image.Image:
    if not text.strip():
        return img
    font = _load_serif_font(font_size)
    w, h = img.size
    tw = _text_width(font, text)
    th = font_size + 6
    cap_h = margin_top + th + margin_bottom
    out = Image.new("RGB", (w, h + cap_h), (255, 255, 255))
    out.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    x = (w - tw) // 2
    y = h + margin_top
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_pdf = root / "resources" / "tsne_2 (2).pdf"
    default_out = root / "resources" / "tsne_ours_with_legend.png"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", type=Path, default=default_pdf)
    p.add_argument("--out", type=Path, default=default_out)
    p.add_argument(
        "--style",
        choices=("paper", "gray-panel"),
        default="paper",
        help="paper = white + serif legend like the figure; gray-panel = old bold band",
    )
    p.add_argument("--raster-dpi", type=int, default=400)
    pg = p.add_mutually_exclusive_group()
    pg.add_argument(
        "--plot-frame",
        dest="plot_frame",
        action="store_true",
        default=True,
        help="Draw a thin black rectangle around the rasterized PDF (default).",
    )
    pg.add_argument(
        "--no-plot-frame",
        dest="plot_frame",
        action="store_false",
        help="Do not draw a frame around the plot.",
    )
    p.add_argument(
        "--caption",
        type=str,
        default="",
        help="Centered serif subcaption below the figure; default empty (omit).",
    )
    p.add_argument("--caption-font-size", type=int, default=15)
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=12.0,
        help="Legend label size (pt)",
    )
    p.add_argument("--legend-dpi", type=int, default=220)
    p.add_argument(
        "--legend-width-frac",
        type=float,
        default=0.48,
        help="Matplotlib legend figure width vs height (wider = less 'thin strip')",
    )
    p.add_argument(
        "--min-legend-width",
        type=int,
        default=320,
        help="Minimum legend column width in pixels (avoids unreadable thin bar)",
    )
    p.add_argument(
        "--legend-image",
        type=Path,
        default=None,
        help="Optional path to a legend/colorbar PNG (or JPEG); pasted right of the "
        "rasterized PDF with height matched to the plot. Skips built-in legend.",
    )
    p.add_argument(
        "--gap",
        type=int,
        default=24,
        help="White space (px) between plot and legend",
    )
    p.add_argument(
        "--legend-frac",
        type=float,
        default=0.46,
        help="(gray-panel) legend width vs plot width",
    )
    p.add_argument("--bar-width", type=int, default=28)
    p.add_argument("--font-size", type=int, default=13, help="(gray-panel)")
    p.add_argument("--pad", type=int, default=12, help="(gray-panel)")
    args = p.parse_args()

    if not args.pdf.is_file():
        print(f"Missing PDF: {args.pdf}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_base = Path(td) / "tsne_raster"
        raster = rasterize_pdf(args.pdf, args.raster_dpi, tmp_base)
        plot = Image.open(raster).convert("RGB")
        if args.plot_frame:
            plot = draw_plot_frame(
                plot,
                width_px=1,
                omit_right=args.legend_image is not None,
            )

        if args.legend_image is not None:
            if not args.legend_image.is_file():
                print(f"Missing --legend-image: {args.legend_image}", file=sys.stderr)
                return 1
            leg_user = Image.open(args.legend_image)
            body = compose_external_legend_paste(plot, leg_user, args.gap)
            note = "external legend image"
        elif args.style == "gray-panel":
            body = compose_embedded_legend(
                plot,
                legend_frac=args.legend_frac,
                pad_px=args.pad,
                bar_w=args.bar_width,
                font_px=args.font_size,
            )
            note = "gray-panel"
        else:
            body = compose_paper_paste(
                plot,
                gap_px=args.gap,
                fontsize=args.legend_fontsize,
                leg_dpi=args.legend_dpi,
                w_frac=args.legend_width_frac,
                min_legend_px=args.min_legend_width,
            )
            note = "paper" + (" (matplotlib legend)" if _HAS_MPL else " (PIL legend)")

        caption = (args.caption or "").strip()
        final = add_subcaption(
            body,
            caption,
            font_size=args.caption_font_size,
        )

        final.save(args.out, format="PNG", optimize=True)

    print(f"Wrote {args.out} (plot {args.raster_dpi} dpi, {note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
