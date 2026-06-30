#!/usr/bin/env python3
"""Remove bottom subcaption (e.g. ``(b) Ours``) from a composite t-SNE PNG.

Upscale first (sharper boundary), detect text in the bottom-center band, crop
above it, then resize back to the **original width** so the asset stays a
reasonable size for README.

Usage:
  python3 tools/crop_tsne_readme_figure.py \\
      -i resources/tsne_ours_with_legend_upload_with_caption.png \\
      -o resources/tsne_ours_with_legend.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def crop_caption(
    img: Image.Image,
    *,
    scale: float,
    bottom_frac: float,
    x_margin_frac: float,
    dark_thresh: float,
    row_dark_min: float,
    margin_px_scaled: int,
    output_width: int | None,
) -> Image.Image:
    """Upscale → locate bottom caption band → crop → optionally downscale width."""
    w0, h0 = img.size
    W = max(1, int(round(w0 * scale)))
    H = max(1, int(round(h0 * scale)))
    up = img.convert("RGB").resize((W, H), _LANCZOS)
    gray = np.asarray(up.convert("L"), dtype=np.float32)

    y0 = int(H * (1.0 - bottom_frac))
    x0 = int(W * x_margin_frac)
    x1 = int(W * (1.0 - x_margin_frac))
    band = gray[y0:H, x0:x1]
    is_dark = band < dark_thresh
    row_dark_frac = is_dark.mean(axis=1)

    idx = np.where(row_dark_frac > row_dark_min)[0]
    if len(idx) == 0:
        cropped = up
    else:
        rel_top = int(idx.min())
        abs_top = y0 + rel_top
        margin = max(1, int(round(margin_px_scaled * scale)))
        crop_h = max(1, abs_top - margin)
        cropped = up.crop((0, 0, W, crop_h))

    if output_width is None:
        output_width = w0
    tw = max(1, output_width)
    th = max(1, int(round(cropped.size[1] * (tw / cropped.size[0]))))
    return cropped.resize((tw, th), _LANCZOS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("resources/tsne_ours_with_legend_upload_with_caption.png"),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("resources/tsne_ours_with_legend.png"),
    )
    p.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Upscale factor before detecting caption (more stable crop).",
    )
    p.add_argument(
        "--bottom-frac",
        type=float,
        default=0.26,
        help="Only search for caption in the bottom this fraction of the image.",
    )
    p.add_argument(
        "--x-margin-frac",
        type=float,
        default=0.16,
        help="Ignore left/right this fraction (legend on the right stays ignored).",
    )
    p.add_argument("--dark-thresh", type=float, default=248.0)
    p.add_argument(
        "--row-dark-min",
        type=float,
        default=0.004,
        help="Min fraction of dark pixels in a row to count as caption ink.",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=6,
        help="Whitespace (in pre-scale pixels) kept above the caption block.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in px (default: same as input image width).",
    )
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Missing input: {args.input}")

    im = Image.open(args.input)
    out = crop_caption(
        im,
        scale=args.scale,
        bottom_frac=args.bottom_frac,
        x_margin_frac=args.x_margin_frac,
        dark_thresh=args.dark_thresh,
        row_dark_min=args.row_dark_min,
        margin_px_scaled=args.margin,
        output_width=args.width if args.width is not None else im.size[0],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output, format="PNG", optimize=True)
    print(f"Wrote {args.output} ({out.size[0]}x{out.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
