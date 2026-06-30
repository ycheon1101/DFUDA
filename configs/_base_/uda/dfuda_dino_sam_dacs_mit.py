# DAFormer + plain DACS — DINO + SAM (512-dim projection)
_base_ = ['dacs.py']
uda = dict(
    use_dino_proto=True,
    use_sam=True,
    proto_proj_in_channels=512,
)
