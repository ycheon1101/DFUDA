# DFUDA: SAM only on fdthings (DeepLabV2 + DAFormer UDA)
_base_ = ['dacs_a999_fdthings.py']
uda = dict(
    use_dino_proto=False,
    use_sam=True,
    sam_start_iter=80000,
    sam_entropy_threshold=0.99,
)
