# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        
        dict(type='WandbLoggerHook',interval=1000,commit=False,
            init_kwargs=dict(
                project='semantic_seg',
                entity = 'bcaitech_cv2',
                name = 'upernet_swin_L_train_all'
            ))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
