import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from utils import SuperpixelSamAutomaticMaskGenerator, labeled_mask, labeled_mask_auto


# ArgumentParser
parser = argparse.ArgumentParser(description="Generate superpixel-guided SAM mask ID maps.")

parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "gta_train",
        "gta5_train",
        "cityscapes_val",
        "cityscapes_train",
        "cityscapes_train_save_with_color",
        "cityscapes_train_automask",
        "synthia_train",
    ],
    default="cityscapes_train",
    help="Dataset split to process.",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device used for SAM inference.")

args = parser.parse_args()


# Paths are resolved from the repository root so the script can run from anywhere.
repo_root = Path(__file__).resolve().parents[1]
data_csv_path = repo_root / "utils" / "data_path"
model_path = repo_root / "pretrained_model"
data_path = repo_root / "data"


def read_image(image_path):
    """Read an image as uint8 RGB, which is required by SAM."""
    image = plt.imread(str(image_path))
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def build_superpixel_generator(sam):
    return SuperpixelSamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=150,
    )


def build_auto_generator(sam):
    return SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=150,
    )


def colorize_mask(mask_with_ids):
    max_id = int(mask_with_ids.max())
    color_map = np.random.randint(0, 256, size=(max_id + 1, 3), dtype=np.uint8)
    return color_map[mask_with_ids]


def get_save_mask_path(img_save_path, img_name, use_cityscapes_format=False):
    if use_cityscapes_format:
        city_name = img_name.split("_")[0]
        return img_save_path / city_name / f"{img_name}_gtFine_labelTrainIds.png"
    return img_save_path / f"{img_name}.png"


def run_mask_generation(
    src_img_path,
    img_save_path,
    mask_generator,
    label_fn,
    save_color=False,
    benchmark=False,
    use_cityscapes_format=False,
):
    img_save_path.mkdir(parents=True, exist_ok=True)

    with src_img_path.open("r", newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        rows = list(csvreader)

    print(f"Input CSV: {src_img_path}")
    print(f"Output directory: {img_save_path}")
    print(f"Number of images: {len(rows)}")


    for index, row in enumerate(rows, start=1):
        img_path = Path(row["src_img"])
        img_name = row.get("img_name") or img_path.stem

        img = read_image(img_path)

        if benchmark and args.device.startswith("cuda"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        mask = mask_generator.generate(img)

        # Label the mask with IDs and boolean mask
        mask_with_ids, boolean_mask = label_fn(mask)
        mask_with_ids = mask_with_ids.astype(np.uint8)

        output = colorize_mask(mask_with_ids) if save_color else mask_with_ids
        save_mask_path = get_save_mask_path(img_save_path, img_name, use_cityscapes_format)
        save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_mask_path), output)

        print(f"[{index}/{len(rows)}] Saved mask for {img_name}: {save_mask_path}")

       

# set SAM
device = args.device
sam_checkpoint = model_path / "SAM" / "sam_vit_h_4b8939.pth"
model_type = "vit_h"

print(f"Dataset: {args.dataset}")
print(f"SAM checkpoint: {sam_checkpoint}")
print(f"Device: {device}")

sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
sam.to(device=device)
sam.eval()

if device.startswith("cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


# gta_train
if args.dataset in ["gta_train", "gta5_train"]:
    img_save_path = data_path / "gta" / "gta_sam_mask_superpixel"
    src_img_path = data_csv_path / "gta_train_paths.csv"
    mask_generator = build_superpixel_generator(sam)

    print("Mask generator: Superpixel-guided SAM")
    print("Mask labeling: labeled_mask")
    run_mask_generation(src_img_path, img_save_path, mask_generator, labeled_mask)


# cityscapes_val
if args.dataset == "cityscapes_val":
    img_save_path = data_path / "cityscapes" / "cityscapes_val_sam_mask_superpixel"
    src_img_path = data_csv_path / "cityscapes_val_paths.csv"
    mask_generator = build_superpixel_generator(sam)

    print("Mask generator: Superpixel-guided SAM")
    print("Mask labeling: labeled_mask")
    run_mask_generation(src_img_path, img_save_path, mask_generator, labeled_mask, use_cityscapes_format=True)


# cityscapes_train
if args.dataset == "cityscapes_train":
    img_save_path = data_path / "cityscapes" / "cityscapes_train_sam_mask_superpixel"
    src_img_path = data_csv_path / "cityscapes_train_paths.csv"
    mask_generator = build_superpixel_generator(sam)

    print("Mask generator: Superpixel-guided SAM")
    print("Mask labeling: labeled_mask")
    run_mask_generation(
        src_img_path,
        img_save_path,
        mask_generator,
        labeled_mask,
        benchmark=True,
        use_cityscapes_format=True,
    )


# cityscapes_train_automask
if args.dataset == "cityscapes_train_automask":
    img_save_path = data_path / "cityscapes" / "cityscapes_train_sam_mask_superpixel_auto"
    src_img_path = data_csv_path / "cityscapes_train_paths.csv"
    mask_generator = build_auto_generator(sam)

    print("Mask generator: SAM automatic mask generator")
    print("Mask labeling: labeled_mask_auto")
    run_mask_generation(src_img_path, img_save_path, mask_generator, labeled_mask_auto, use_cityscapes_format=True)


# cityscapes_train_save_with_color
if args.dataset == "cityscapes_train_save_with_color":
    img_save_path = data_path / "cityscapes" / "cityscapes_train_sam_mask_superpixel_color"
    src_img_path = data_csv_path / "cityscapes_train_paths.csv"
    mask_generator = build_superpixel_generator(sam)

    print("Mask generator: Superpixel-guided SAM")
    print("Mask labeling: labeled_mask")
    run_mask_generation(
        src_img_path,
        img_save_path,
        mask_generator,
        labeled_mask,
        save_color=True,
        use_cityscapes_format=True,
    )


# synthia_train
if args.dataset == "synthia_train":
    img_save_path = data_path / "synthia" / "synthia_sam_mask_superpixel"
    src_img_path = data_csv_path / "synthia_train_paths.csv"
    mask_generator = build_superpixel_generator(sam)

    print("Mask generator: Superpixel-guided SAM")
    print("Mask labeling: labeled_mask")
    run_mask_generation(src_img_path, img_save_path, mask_generator, labeled_mask)

