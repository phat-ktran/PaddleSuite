from object_detection.predictor import ObjectDetector

# 1. Initialize
detector = ObjectDetector(
    model_dir="models/",
    device="GPU",
)

results = detector.predict("data/images/")
