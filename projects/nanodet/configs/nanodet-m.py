_base_ = [
    '../../../configs/_base_/datasets/coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
max_epochs = 280

custom_imports=dict(imports=[
    'mmcls.models',
    'projects.nanodet.nanodet',
], allow_failed_imports=False) 

model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='mmcls.ShuffleNetV2',
        widen_factor=1.0, # ShuffleNetV2 x1.0
        out_indices=(2, 3, 4),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type="LeakyReLU"),
        norm_eval=False,
        with_cp=False,
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[116,232,464],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='nanodet.NanoDetHead',
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        share_cls_reg=True,
        norm_cfg=dict(
            type="BN",
            requires_grad=True,
        ),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=5,
            scales_per_octave=1,
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=7,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.14, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[240,260,275],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=10)


train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomAffine',
        max_rotate_degree=0,
        max_translate_ratio=0.2,
        max_shear_degree=0,
        scaling_ratio_range=[0.6,1.4],
        # TODO: 缺失stretch增强
    ),
    dict(
        type='RandomFlip',
        prob=0.5
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=51,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.5, 1.2),
        hue_delta=0
    ),
    dict(
        type='Resize', scale=(320, 320),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=192,
    num_workers=8,
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline)
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args={_base_.file_client_args}),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]