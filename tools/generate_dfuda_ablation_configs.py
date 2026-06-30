#!/usr/bin/env python3
"""Generate Table-3 ablation configs: 8 rows x 4 components = 32 configs."""
import os

ROOT = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dfuda', 'ablation')
os.makedirs(ROOT, exist_ok=True)

ITERS = 300000
EVAL_INTERVAL = 4000

STACKS = {
    'dlv2_dacs': dict(
        architecture='dlv2red', crop='512x512', hrda=False, mic=False,
        rcs_T=None, plcrop=False, inference='whole', sync_crop_size=None,
        rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs',
            'sam': 'dfuda_sam_dacs',
            'dino': 'dfuda_dino_dacs',
            'full': 'dfuda_dino_sam_dacs',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'dlv2_daformer': dict(
        architecture='dlv2red', crop='512x512', hrda=False, mic=False,
        rcs_T=0.01, plcrop=True, inference='whole', sync_crop_size=None,
        rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam_fdthings',
            'dino': 'dfuda_dino_fdthings',
            'full': 'dfuda_dino_sam_fdthings',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'dlv2_hrda': dict(
        architecture='hrda1-512-0.1_dlv2red', crop='512x512', hrda=True,
        mic=False, rcs_T=0.01, plcrop='v2', inference='whole',
        sync_crop_size=None, rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam_fdthings',
            'dino': 'dfuda_dino_fdthings',
            'full': 'dfuda_dino_sam_fdthings',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'dlv2_mic': dict(
        architecture='hrda1-512-0.1_dlv2red', crop='512x512', hrda=True,
        mic=True, rcs_T=0.01, plcrop='v2', inference='whole',
        sync_crop_size=None, rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam_fdthings',
            'dino': 'dfuda_dino_fdthings',
            'full': 'dfuda_dino_sam_fdthings',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'daformer_dacs': dict(
        architecture='daformer_sepaspp', crop='512x512', hrda=False, mic=False,
        rcs_T=None, plcrop=False, inference='whole', sync_crop_size=None,
        rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs',
            'sam': 'dfuda_sam_dacs_mit',
            'dino': 'dfuda_dino_dacs_mit',
            'full': 'dfuda_dino_sam_dacs_mit',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'daformer_daformer': dict(
        architecture='daformer_sepaspp', crop='512x512', hrda=False, mic=False,
        rcs_T=0.01, plcrop=True, inference='whole', sync_crop_size=None,
        rcs_min_crop=0.5,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam',
            'dino': 'dfuda_dino',
            'full': 'dfuda_dino_sam',
        },
        dataset='uda_gta_to_cityscapes_512x512',
    ),
    'daformer_hrda': dict(
        architecture='hrda1-512-0.1_daformer_sepaspp', crop='1024x1024',
        hrda=True, mic=False, rcs_T=0.01, plcrop='v2', inference='slide',
        sync_crop_size=(512, 512), rcs_min_crop=2.0,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam',
            'dino': 'dfuda_dino',
            'full': 'dfuda_dino_sam',
        },
        dataset='uda_gtaHR_to_cityscapesHR_1024x1024',
    ),
    'daformer_mic': dict(
        architecture='hrda1-512-0.1_daformer_sepaspp', crop='1024x1024',
        hrda=True, mic=True, rcs_T=0.01, plcrop='v2', inference='slide',
        sync_crop_size=(512, 512), rcs_min_crop=2.0,
        uda_map={
            'baseline': 'dacs_a999_fdthings',
            'sam': 'dfuda_sam',
            'dino': 'dfuda_dino',
            'full': 'dfuda_dino_sam',
        },
        dataset='uda_gtaHR_to_cityscapesHR_1024x1024',
    ),
}

SAM_COMPONENTS = {'sam', 'full'}

for stack_key, s in STACKS.items():
    for component in ['baseline', 'sam', 'dino', 'full']:
        fname = f'gta2cs_{stack_key}_{component}.py'
        uda_cfg = s['uda_map'][component]
        dataset = s['dataset'] + (
            '_with_sam' if component in SAM_COMPONENTS else '')
        name = f'gta2cs_{stack_key}_{component}'
        lines = [
            f"# Table 3 row `{stack_key}`: {component}",
            '_base_ = [',
            "    '../../_base_/default_runtime.py',",
        ]
        if 'dlv2' in s['architecture']:
            lines.append("    '../../_base_/models/deeplabv2red_r50-d8.py',")
        else:
            lines.append("    '../../_base_/models/daformer_sepaspp_mitb5.py',")
        lines.append(f"    '../../_base_/datasets/{dataset}.py',")
        lines.append(f"    '../../_base_/uda/{uda_cfg}.py',")
        lines.append("    '../../_base_/schedules/adamw.py',")
        lines.append("    '../../_base_/schedules/poly10warm.py',")
        lines.append(']')
        lines += ['seed = 0', 'n_gpus = 1', "gpu_model = 'NVIDIATITANRTX'",
                  f"name = '{name}'", "exp = 'dfuda_ablation'"]
        if s['hrda']:
            cr = s['crop'].split('x')
            test_cfg = ''
            if s['inference'] == 'slide':
                test_cfg = (
                    f", test_cfg=dict(mode='slide', batched_slide=True, "
                    f"stride=[{int(cr[0]) // 2}, {int(cr[1]) // 2}], "
                    f"crop_size=[{cr[0]}, {cr[1]}])")
            lines += [
                'model = dict(',
                "    type='HRDAEncoderDecoder',",
                '    decode_head=dict(',
                "        type='HRDAHead',",
                "        single_scale_head='DAFormerHead',",
                '        attention_classwise=True,',
                '        hr_loss_weight=0.1),',
                '    scales=[1, 0.5],',
                '    hr_crop_size=(512, 512),',
                '    feature_scale=0.5,',
                '    crop_coord_divisible=8,',
                '    hr_slide_inference=True' + test_cfg + ')',
            ]
        lines += ['data = dict(', '    samples_per_gpu=2,', '    workers_per_gpu=1,',
                  '    train=dict(']
        if s['rcs_T'] is not None:
            lines.append(
                f"        rare_class_sampling=dict(min_pixels=3000, "
                f"class_temp={s['rcs_T']}, min_crop_ratio={s['rcs_min_crop']}),")
        if s['plcrop'] == 'v2':
            lines.append(
                '        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),')
        if s['sync_crop_size']:
            lines.append(f"        sync_crop_size={s['sync_crop_size']},")
        lines += ['    ),', ')']
        if s['mic']:
            lines += [
                'uda = dict(',
                "    mask_mode='separatetrgaug',",
                "    mask_alpha='same',",
                "    mask_pseudo_threshold='same',",
                '    mask_lambda=1,',
                '    mask_generator=dict(',
                "        type='block', mask_ratio=0.7, mask_block_size=64, "
                '_delete_=True))',
            ]
        lines += [
            'optimizer_config = None',
            'optimizer = dict(',
            '    lr=6e-05,',
            '    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))',
            f"runner = dict(type='IterBasedRunner', max_iters={ITERS})",
            f'checkpoint_config = dict(by_epoch=False, interval={ITERS}, '
            'max_keep_ckpts=1)',
            f"evaluation = dict(interval={EVAL_INTERVAL}, metric='mIoU')",
            '',
        ]
        with open(os.path.join(ROOT, fname), 'w') as f:
            f.write('\n'.join(lines))

# Remove legacy 3-component-only files if any remain without sam variant
for old in os.listdir(ROOT):
    if old.endswith('.py') and old.count('_') >= 3:
        parts = old.replace('.py', '').split('_')
        if parts[-1] not in ('baseline', 'sam', 'dino', 'full'):
            os.remove(os.path.join(ROOT, old))

print(f'Generated {len(STACKS) * 4} configs in {ROOT}')
