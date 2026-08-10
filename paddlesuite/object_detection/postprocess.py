import cv2
import numpy as np


def nms(boxes, nms_threshold=0.5, score_threshold=0.01):
    """Apply single-class NMS to ``[class_id, score, x1, y1, x2, y2]`` boxes."""
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = boxes[boxes[:, 1] > score_threshold]
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xywh_boxes = boxes[:, 2:6].copy()
    xywh_boxes[:, 2:4] -= xywh_boxes[:, 0:2]
    keep_indices = cv2.dnn.NMSBoxes(
        xywh_boxes,
        boxes[:, 1],
        score_threshold,
        nms_threshold,
    )
    if len(keep_indices) == 0:
        return np.empty((0, 6), dtype=np.float32)

    kept = boxes[np.asarray(keep_indices).reshape(-1)]
    return kept[kept[:, 1].argsort()[::-1]]


def merge_scale_boxes(image_boxes, nms_threshold=0.5, score_threshold=0.01):
    """Merge one image's scale outputs and apply single-class NMS."""
    if not image_boxes:
        return np.empty((0, 6), dtype=np.float32)

    return nms(
        np.concatenate(image_boxes, axis=0),
        nms_threshold=nms_threshold,
        score_threshold=score_threshold,
    )
