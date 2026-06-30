# DFUDA main result: GTA -> Cityscapes (MIC+HRDA + DINO + SAM)
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_gtaHR_to_cityscapesHR_1024x1024_with_sam.py',
    '../_base_/uda/dfuda_dino_sam.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
    '_base_mic_hrda.py',
]
seed = 2
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
name = 'gtaHR2csHR_dfuda'
exp = 'dfuda'
name_dataset = 'gtaHR2cityscapesHR_1024x1024'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dfuda_dino_sam_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
