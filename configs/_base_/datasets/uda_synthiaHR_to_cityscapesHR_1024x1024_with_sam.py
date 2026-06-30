# SAM superpixel masks for SYNTHIA->Cityscapes HR training
_base_ = ['uda_synthiaHR_to_cityscapesHR_1024x1024.py']
data = dict(
    train=dict(
        target=dict(
            ann_dir='cityscapes_train_sam_mask_superpixel',
        ),
    ),
)
