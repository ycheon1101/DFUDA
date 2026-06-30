# GTA→Cityscapes with SAM superpixel masks on the target domain
_base_ = ['uda_gta_to_cityscapes_512x512.py']

data = dict(
    train=dict(
        target=dict(
            ann_dir='cityscapes_train_sam_mask_superpixel',
        ),
    ),
)
