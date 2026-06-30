# Table 3 row `daformer_hrda`: baseline
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/daformer_sepaspp_mitb5.py',
    '../../_base_/datasets/uda_gtaHR_to_cityscapesHR_1024x1024.py',
    '../../_base_/uda/dacs_a999_fdthings.py',
    '../../_base_/schedules/adamw.py',
    '../../_base_/schedules/poly10warm.py',
]
seed = 0
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
name = 'gta2cs_daformer_hrda_baseline'
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
    hr_slide_inference=True, test_cfg=dict(mode='slide', batched_slide=True, stride=[512, 512], crop_size=[1024, 1024]))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        rare_class_sampling=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
        sync_crop_size=(512, 512),
    ),
)
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=300000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
