import argparse
import csv
from pathlib import Path
from typing import List

# Utility script to generate dataset CSV files for the SAM mask generation pipeline.
# Default output format is lightweight (img_name, src_img), with optional GT column.

VALID_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def sort_key(path: Path) -> int:
    """Sort file names by the numeric part (e.g., 2.png before 10.png)."""
    return int("".join(filter(str.isdigit, path.name)) or 0)


def list_images(directory: Path) -> List[Path]:
    """Return supported image files in deterministic numeric order."""
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES],
        key=sort_key,
    )


def collect_gta_rows(data_root: Path, include_gt: bool, gt_type: str = "color") -> List[List[str]]:
    """Collect rows for GTA with optional label paths."""
    images_dir = data_root / "gta" / "images"
    labels_dir = data_root / "gta" / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {images_dir}")
    if include_gt and not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {labels_dir}")

    rows = []
    for image_path in list_images(images_dir):
        row = [image_path.stem, str(image_path)]
        if include_gt:
            if gt_type == "trainid":
                label_path = labels_dir / f"{image_path.stem}_labelTrainIds.png"
            else:
                label_path = labels_dir / image_path.name
            if not label_path.exists():
                continue
            row.append(str(label_path))
        rows.append(row)
    return rows


def collect_cityscapes_rows(data_root: Path, split: str, include_gt: bool) -> List[List[str]]:
    """Collect rows for Cityscapes split with optional labelIds paths."""
    left_dir = data_root / "cityscapes" / "leftImg8bit" / split
    gt_dir = data_root / "cityscapes" / "gtFine" / split
    if not left_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {left_dir}")
    if include_gt and not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {gt_dir}")

    rows = []
    for city_dir in sorted([p for p in left_dir.iterdir() if p.is_dir()]):
        gt_city_dir = gt_dir / city_dir.name if include_gt else None
        if include_gt and not gt_city_dir.is_dir():
            continue

        for image_path in list_images(city_dir):
            # Convert "..._leftImg8bit.png" -> base image id used in CSV.
            img_name = image_path.name.replace("_leftImg8bit", "").rsplit(".", 1)[0]
            row = [img_name, str(image_path)]
            if include_gt:
                # Cityscapes GT naming convention:
                # <base>_leftImg8bit.png -> <base>_gtFine_labelIds.png
                gt_path = gt_city_dir / image_path.name.replace(
                    "_leftImg8bit.png", "_gtFine_labelIds.png"
                )
                if not gt_path.exists():
                    continue
                row.append(str(gt_path))
            rows.append(row)
    return rows


def collect_synthia_rows(data_root: Path, include_gt: bool) -> List[List[str]]:
    """Collect rows for SYNTHIA with optional GT label paths."""
    rgb_dir = data_root / "synthia" / "RGB"
    gt_dir = data_root / "synthia" / "GT" / "LABELS"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {rgb_dir}")
    if include_gt and not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {gt_dir}")

    rows = []
    for image_path in list_images(rgb_dir):
        row = [image_path.stem, str(image_path)]
        if include_gt:
            gt_path = gt_dir / image_path.name
            if not gt_path.exists():
                continue
            row.append(str(gt_path))
        rows.append(row)
    return rows


def write_csv(output_csv: Path, header: List[str], rows: List[List[str]]) -> None:
    """Write header + rows to CSV, creating the parent directory if needed."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Define CLI arguments for dataset CSV generation."""
    parser = argparse.ArgumentParser(description="Create dataset CSV from local DFUDA data layout.")
    parser.add_argument(
        "--dataset",
        choices=["gta", "cityscapes", "synthia"],
        required=True,
        help="Dataset to index.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Cityscapes split to use (only for --dataset cityscapes).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <repo>/utils/<dataset>_<split>_paths.csv.",
    )
    parser.add_argument(
        "--include-gt",
        action="store_true",
        help="Include GT path column in CSV.",
    )
    parser.add_argument(
        "--gt-type",
        choices=["color", "trainid"],
        default="color",
        help="GTA label type when --include-gt is set: color (raw labels) or "
        "trainid (_labelTrainIds.png, requires tools/convert_datasets/gta.py).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for building dataset CSV files."""
    args = parse_args()
    default_name = f"{args.dataset}_{args.split}_paths.csv"
    # Default output directory requested by project convention.
    output_path = args.output or (Path(__file__).resolve().parent / "data_path" / default_name)

    if args.dataset == "gta":
        rows = collect_gta_rows(args.data_root, args.include_gt, args.gt_type)
        if args.include_gt:
            gt_column = "gt_trainid" if args.gt_type == "trainid" else "gt_color"
            header = ["img_name", "src_img", gt_column]
        else:
            header = ["img_name", "src_img"]
    elif args.dataset == "cityscapes":
        rows = collect_cityscapes_rows(args.data_root, args.split, args.include_gt)
        header = ["img_name", "src_img"] + (["gt_id"] if args.include_gt else [])
    else:
        rows = collect_synthia_rows(args.data_root, args.include_gt)
        header = ["img_name", "src_img"] + (["gt_id"] if args.include_gt else [])

    write_csv(output_path, header, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

