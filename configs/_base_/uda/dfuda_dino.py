# DAFormer / MiT-B5 — DINO prototype loss only
_base_ = ['dacs_a999_fdthings.py']
uda = dict(
    use_dino_proto=True,
    use_sam=False,
    proto_proj_in_channels=512,
)
