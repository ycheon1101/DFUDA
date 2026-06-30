# DeepLabV2 / ResNet — DINO + SAM pseudo-label refinement
_base_ = ['dacs.py']
uda = dict(
    use_dino_proto=True,
    use_sam=True,
    proto_proj_in_channels=2048,
)
