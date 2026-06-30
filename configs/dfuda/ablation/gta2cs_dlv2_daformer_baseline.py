# Table 3 row `dlv2_daformer`: baseline
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv2red_r50-d8.py',
    '../../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../../_base_/uda/dacs_a999_fdthings.py',
    '../../_base_/schedules/adamw.py',
    '../../_base_/schedules/poly10warm.py',
]
seed = 0
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
name = 'gta2cs_dlv2_daformer_baseline'
exp = 'dfuda_ablation'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        rare_class_sampling=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
    ),
)
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=300000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
