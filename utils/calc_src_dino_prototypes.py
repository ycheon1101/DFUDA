import argparse
import csv
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV3_GITHUB_LOCATION = str(REPO_ROOT / "dinov3")
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_NAME = MODEL_DINOV3_VITL
PATCH_SIZE = 16
IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT = 1280, 720

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

GTA_CSV_PATH = REPO_ROOT / "utils" / "data_path" / "gta_train_paths.csv"
PROTOTYPE_DIR = REPO_ROOT / "prototypes"

# TODO: Set this to your local ViT-L/16 distilled checkpoint path after download.
# Example: /home/DFUDA/pretrained_model/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
DINOV3_WEIGHT_PATH = "/home/DFUDA/pretrained_model/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

CLASS_IDS = list(range(19))
NUM_CLASSES = len(CLASS_IDS)

CITYSCAPES_TRAINID_TO_NAME = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "ignore",
}

MODE_CONFIG = {
    "orig": {
        "use_aug": [False],
        "output_name": "prototypes_19classes_orig.pt",
        "checkpoint_name": "proto_running_ckpt_orig.npz",
        "description": "original images only",
    },
    "aug": {
        "use_aug": [True],
        "output_name": "prototypes_19classes_aug.pt",
        "checkpoint_name": "proto_running_ckpt_aug.npz",
        "description": "photometrically augmented images only",
    },
    "orig_aug": {
        "use_aug": [False, True],
        "output_name": "prototypes_19classes_orig_aug.pt",
        "checkpoint_name": "proto_running_ckpt_orig_aug.npz",
        "description": "original and augmented images",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build DINOv3 class prototypes from GTA labels.")
    parser.add_argument(
        "--mode",
        choices=list(MODE_CONFIG.keys()),
        default="orig",
        help="Prototype variant to generate.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=GTA_CSV_PATH,
        help="CSV with src_img and gt_trainid columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROTOTYPE_DIR,
        help="Directory where prototype files are saved.",
    )
    return parser.parse_args()


def resize_transform(pil_img, image_size_h, image_size_w, is_label, patch_size=PATCH_SIZE):
    h_patches = image_size_h // patch_size
    w_patches = image_size_w // patch_size
    new_size = (h_patches * patch_size, w_patches * patch_size)
    if not is_label:
        return TF.to_tensor(TF.resize(pil_img, new_size, interpolation=InterpolationMode.BILINEAR))
    lab = TF.resize(pil_img, new_size, interpolation=InterpolationMode.NEAREST)
    lab_np = np.array(lab, dtype=np.int64)
    return torch.from_numpy(lab_np)


def resize_label_like_image(pil_lab, image_size_h, image_size_w, patch_size=PATCH_SIZE):
    h_patches = image_size_h // patch_size
    w_patches = image_size_w // patch_size
    new_size = (h_patches * patch_size, w_patches * patch_size)
    lab_resized = TF.resize(pil_lab, new_size, interpolation=InterpolationMode.NEAREST)
    return torch.from_numpy(np.array(lab_resized, dtype=np.int64))


def extract_feat_map(pil_img, model, n_layers):
    img = resize_transform(pil_img, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, is_label=False)
    img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img = img.unsqueeze(0).cuda()

    with torch.inference_mode():
        feats_list = model.get_intermediate_layers(img, n=range(n_layers), reshape=True, norm=True)
        feat = feats_list[-1].squeeze(0)

    c, _, _ = feat.shape
    feat = F.normalize(feat.view(c, -1), p=2, dim=0).view(c, feat.shape[1], feat.shape[2])
    return feat


def downsample_label_to_feat_grid(label_tensor_hw, hf, wf):
    lab = label_tensor_hw.long().unsqueeze(0).unsqueeze(0).float()
    lab_ds = F.interpolate(lab, size=(hf, wf), mode="nearest")
    return lab_ds[0, 0].long()


def load_csv_rows(csv_path):
    items = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            items.append((row["src_img"], row["gt_trainid"]))
    return items


def photometric_aug(pil_img):
    img = pil_img
    if random.random() < 0.8:
        b, c, s, h = 0.3, 0.3, 0.3, 0.05
        img = TF.adjust_brightness(img, 1.0 + random.uniform(-b, b))
        img = TF.adjust_contrast(img, 1.0 + random.uniform(-c, c))
        img = TF.adjust_saturation(img, 1.0 + random.uniform(-s, s))
        img = TF.adjust_hue(img, random.uniform(-h, h))
    if random.random() < 0.2:
        img = TF.gaussian_blur(img, kernel_size=3)
    if random.random() < 0.1:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    if random.random() < 0.5:
        g = random.uniform(0.9, 1.1)
        t = TF.to_tensor(img).clamp(0, 1) ** g
        img = TF.to_pil_image(t)
    return img


def build_work_rows(rows, use_aug_flags):
    work_rows = []
    for img_path, gt_path in rows:
        for do_aug in use_aug_flags:
            work_rows.append((img_path, gt_path, do_aug))
    return work_rows


def load_dinov3_model(weight_path):
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {weight_path}")

    has_hub_hash = bool(re.search(r"-[0-9a-f]{8}\.pth$", weight_path.name))
    if has_hub_hash:
        return torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model=MODEL_NAME,
            source="local",
            weights=str(weight_path.resolve()),
        )

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        pretrained=False,
    )
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def main():
    args = parse_args()
    mode_cfg = MODE_CONFIG[args.mode]
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

    if DINOV3_WEIGHT_PATH == "TODO":
        raise ValueError(
            "Set DINOV3_WEIGHT_PATH in utils/calc_src_dino_prototypes.py to your local "
            "ViT-L/16 distilled checkpoint path."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / mode_cfg["checkpoint_name"]
    final_path_pt = args.output_dir / mode_cfg["output_name"]

    print(f"DINOv3 location set to {DINOV3_LOCATION}")
    print(f"Prototype mode: {args.mode} ({mode_cfg['description']})")
    print(f"Output directory: {args.output_dir}")

    model = load_dinov3_model(DINOV3_WEIGHT_PATH)
    print(model.cuda())

    rows = load_csv_rows(args.csv_path)
    work_rows = build_work_rows(rows, mode_cfg["use_aug"])
    total = len(work_rows)

    first_img = Image.open(rows[0][0]).convert("RGB")
    feat0 = extract_feat_map(first_img, model, n_layers)
    c = feat0.shape[0]
    del feat0, first_img
    torch.cuda.empty_cache()

    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        sums = ckpt["sums"].copy()
        counts = ckpt["counts"].copy()
        start_idx = int(ckpt.get("next_index", 0))
        print(f"[Resume] Loaded checkpoint at global index {start_idx}/{total}.")
    else:
        sums = np.zeros((c, NUM_CLASSES), dtype=np.float64)
        counts = np.zeros((NUM_CLASSES,), dtype=np.int64)
        start_idx = 0

    save_every = 1000

    for gidx in tqdm(range(start_idx, total), desc=f"Computing prototypes ({args.mode})"):
        img_path, gt_path, do_aug = work_rows[gidx]

        try:
            with Image.open(img_path) as img_raw:
                img_rgb = img_raw.convert("RGB")
                pil_img = photometric_aug(img_rgb) if do_aug else img_rgb.copy()
        except Exception as exc:
            print(f"[Warn] Failed to open image: {img_path} ({exc}); skip")
            continue

        feat = extract_feat_map(pil_img, model, n_layers)
        c_, hf, wf = feat.shape
        assert c_ == c
        pil_img.close()

        try:
            pil_lab = Image.open(gt_path)
            lab_np = np.array(pil_lab, dtype=np.int64)
            lab_t = torch.from_numpy(lab_np)
        except Exception as exc:
            print(f"[Warn] Failed to open label: {gt_path} ({exc}); skip")
            del feat
            torch.cuda.empty_cache()
            continue

        lab_resized = resize_label_like_image(pil_lab, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, PATCH_SIZE)
        pil_lab.close()
        lab_ds = downsample_label_to_feat_grid(lab_resized, hf, wf)

        feat_cpu = feat.detach().cpu().numpy().reshape(c, -1)
        lab_flat = lab_ds.view(-1).cpu().numpy()

        for local_idx, cls_id in enumerate(CLASS_IDS):
            mask = lab_flat == cls_id
            if not np.any(mask):
                continue
            sums[:, local_idx] += feat_cpu[:, mask].sum(axis=1)
            counts[local_idx] += int(mask.sum())

        del feat, feat_cpu, lab_t, lab_ds
        torch.cuda.empty_cache()

        if (gidx + 1) % save_every == 0:
            np.savez(ckpt_path, sums=sums, counts=counts, next_index=gidx + 1)
            seen = counts.sum()
            print(f"[CKPT] Saved at {gidx + 1}/{total}, total labeled pixels = {seen}")

    np.savez(ckpt_path, sums=sums, counts=counts, next_index=total)

    sums_t = torch.from_numpy(sums).to(torch.float32)
    counts_t = torch.from_numpy(counts).to(torch.float32)

    protos_t = torch.zeros((c, NUM_CLASSES), dtype=torch.float32)
    for j in range(NUM_CLASSES):
        if counts_t[j] > 0:
            proto = sums_t[:, j] / counts_t[j]
            proto = F.normalize(proto.unsqueeze(0), p=2, dim=1, eps=1e-12)
            protos_t[:, j] = proto.squeeze(0)
        else:
            protos_t[:, j] = 0.0

    torch.save(
        {
            "prototypes": protos_t,
            "class_ids": torch.tensor(CLASS_IDS, dtype=torch.long),
            "class_names": [CITYSCAPES_TRAINID_TO_NAME[cid] for cid in CLASS_IDS],
            "mode": args.mode,
            "sums": sums_t,
            "counts": counts_t,
        },
        final_path_pt,
    )

    print(f"[DONE] Saved final prototypes to: {final_path_pt}")
    print(f"[INFO] You can resume later using checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
