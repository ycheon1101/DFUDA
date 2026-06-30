# GTA-HR→Cityscapes-HR with SAM superpixel masks on the target domain
_base_ = ['uda_gtaHR_to_cityscapesHR_1024x1024.py']

data = dict(
    train=dict(
        target=dict(
            ann_dir='cityscapes_train_sam_mask_superpixel',
        ),
    ),
)
