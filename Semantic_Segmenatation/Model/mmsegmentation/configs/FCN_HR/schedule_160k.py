# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.25,
    by_epoch=False,
    periods=[5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 1700],
    restart_weights=[1, 1, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.075, 0.075, 0.05, 0.05],
    min_lr=1e-6)

# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)#16000
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)#16000
