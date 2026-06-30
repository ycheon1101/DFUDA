# DeepLabV2 + fdthings — DINO + SAM
_base_ = ['dacs_a999_fdthings.py']
uda = dict(
    use_dino_proto=True,
    use_sam=True,
    proto_proj_in_channels=2048,
)
