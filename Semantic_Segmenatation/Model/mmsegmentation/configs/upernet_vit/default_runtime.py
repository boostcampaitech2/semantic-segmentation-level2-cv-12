# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',interval=100,
            commit=False,
            init_kwargs=dict(
                project='semantic_seg',
                entity = 'bcaitech_cv2',
                name = 'upernet_vit_ln_mln'
            ))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True