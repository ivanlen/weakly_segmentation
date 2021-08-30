import numpy as np


def miou(image_preds, image_seg_classes, n_classes=5):
    """
    computes the mean Intersection over Union score for an image and the pixel predictions.

    some refs:
    - https://medium.com/@cyborg.team.nitr/miou-calculation-4875f918f4cb
    - https://github.com/CYBORG-NIT-ROURKELA/Improving_Semantic_segmentation/blob/master/miou_calculation.py
    :return: float: mIoU score
    """
    assert image_preds.shape == image_seg_classes.shape, 'gt and image shapes must be equal'
    preds = np.ravel(image_preds)
    gt = np.ravel(image_seg_classes)


def iou(pred, target, n_classes = 12):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
        ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
        ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

