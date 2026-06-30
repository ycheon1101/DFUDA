# Table 3 row `dlv2_mic`: full
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv2red_r50-d8.py',
    '../../_base_/datasets/uda_gta_to_cityscapes_512x512_with_sam.py',
    '../../_base_/uda/dfuda_dino_sam_fdthings.py',
    '../../_base_/schedules/adamw.py',
    '../../_base_/schedules/poly10warm.py',
]
seed = 0
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
name = 'gta2cs_dlv2_mic_full'
exp = 'dfuda_ablation'
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        rare_class_sampling=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
)
uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=300000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
