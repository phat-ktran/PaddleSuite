from paddlesuite.object_detection.predictor import ObjectDetector

# 1. Initialize
detector = ObjectDetector(
    model_dir="/home/phatalways_sleeping/Documents/university/thesis/src/ConsensusOCR/models/faster_rcnn_r50_fpn_1x_hisdoc1b",
    device="GPU",
)

results = detector.predict("data/", batch_size=4)

print(results[0])
