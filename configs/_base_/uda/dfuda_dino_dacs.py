# DeepLabV2 / ResNet — DINO prototype loss only
_base_ = ['dacs.py']
uda = dict(
    use_dino_proto=True,
    use_sam=False,
    proto_proj_in_channels=2048,
)
