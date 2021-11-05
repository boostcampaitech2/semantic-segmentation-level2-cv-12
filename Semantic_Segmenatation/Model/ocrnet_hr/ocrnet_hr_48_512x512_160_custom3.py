# _base_ = './ocrnet_hr18_512x512_160k_ade20k.py'
_base_ = './ocrnet_hr_512x512_custom3.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    # pretrained='open-mmlab://msra/hrnetv2_w48',
    pretrained = '/opt/ml/segmentation/HRNet_W48_C_ssld_pretrained.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=11, # 150 -> 11
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                class_weight = [0.732516,1.442744,0.865295,1.156541,1.120115,1.099340,1.370468,0.890204,0.817555,0.787677,1.293223])),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=11, # 150 -> 11
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                class_weight = [0.732516,1.442744,0.865295,1.156541,1.120115,1.099340,1.370468,0.890204,0.817555,0.787677,1.293223])),
    ])
