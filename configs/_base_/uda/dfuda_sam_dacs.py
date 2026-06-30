# DFUDA: SAM only on plain DACS (DeepLabV2 / ResNet)
_base_ = ['dacs.py']
uda = dict(
    use_dino_proto=False,
    use_sam=True,
    sam_start_iter=80000,
    sam_entropy_threshold=0.99,
)
