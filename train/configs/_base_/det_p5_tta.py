tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))

img_scales = [(1280, 1280), (960, 960), (1600, 1600)]

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='YOLOv5KeepRatioResize', scale=s) for s in img_scales],
            [
                dict(
                    type='LetterResize',
                    scale=s,
                    allow_scale_up=False,
                    pad_val=dict(img=114)) for s in img_scales
            ],
            [dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet')],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param'))
            ],
        ])
]
