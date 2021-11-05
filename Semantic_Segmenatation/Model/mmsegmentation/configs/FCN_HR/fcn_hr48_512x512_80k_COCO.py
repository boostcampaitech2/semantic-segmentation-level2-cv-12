_base_ = [
    'fcn_hr18.py', 'custom_dataset.py',
    'default_runtime.py', 'schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=11))
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
