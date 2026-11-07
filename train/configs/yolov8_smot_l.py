"""YOLOv8-L config for SMOT4SB SliceTrain.

This is the primary config corresponding to the DL Team's winning model
(43.7M params, SO-HOTA 55.205).
"""

_base_ = './yolov8_smot_m.py'

deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

pre_transform = _base_.pre_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])))
