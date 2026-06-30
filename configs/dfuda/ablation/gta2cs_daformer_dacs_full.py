# Table 3 row `daformer_dacs`: full
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/daformer_sepaspp_mitb5.py',
    '../../_base_/datasets/uda_gta_to_cityscapes_512x512_with_sam.py',
    '../../_base_/uda/dfuda_dino_sam_dacs_mit.py',
    '../../_base_/schedules/adamw.py',
    '../../_base_/schedules/poly10warm.py',
]
seed = 0
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
name = 'gta2cs_daformer_dacs_full'
exp = 'dfuda_ablation'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
    ),
)
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=300000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
