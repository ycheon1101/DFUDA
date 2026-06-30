# DAFormer + plain DACS — DINO prototype loss only (512-dim projection)
_base_ = ['dacs.py']
uda = dict(
    use_dino_proto=True,
    use_sam=False,
    proto_proj_in_channels=512,
)
