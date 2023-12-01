# The new config inherits a base config to highlight the necessary modification

_base_ = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/mmdetection_base/mmdetection/configs/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4), mask_head=False))

# Modify dataset related settings
data_root = ''
metainfo = {
    'classes': ('nonTIL_stromal', 'sTIL', 'tumor_any', 'other', ),
}
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection/fold_1/train.json')
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection/fold_1/val.json')
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        test_mode=True,
        ann_file=f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection/fold_1/test.json')
)

# Modify metric related settings
val_evaluator = dict(ann_file=f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection/fold_1/val.json')
test_evaluator = dict(ann_file=f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection/fold_1/test.json')

train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=5,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
