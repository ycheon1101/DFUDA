# GTA→Cityscapes with optional SAM superpixel masks (test / partial masks)
_base_ = ['uda_gta_to_cityscapes_512x512.py']

cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', optional=True),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    train=dict(
        skip_missing_sam_mask=True,
        target=dict(
            ann_dir='cityscapes_train_sam_mask_superpixel',
            pipeline=cityscapes_train_pipeline,
        ),
    ),
)
