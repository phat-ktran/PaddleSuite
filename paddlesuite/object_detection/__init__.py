from typing import TYPE_CHECKING

from .postprocess import merge_scale_boxes, nms

if TYPE_CHECKING:
    from .predictor import ObjectDetector
    from .result import ObjectDetectorResult

__all__ = [
    "ObjectDetector",
    "ObjectDetectorResult",
    "merge_scale_boxes",
    "nms",
]


def __getattr__(name):
    if name == "ObjectDetector":
        from .predictor import ObjectDetector

        return ObjectDetector
    if name == "ObjectDetectorResult":
        from .result import ObjectDetectorResult

        return ObjectDetectorResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
