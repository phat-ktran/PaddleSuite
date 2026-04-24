from classification.predictor import ImageClassifier

# 1. Initialize
detector = ImageClassifier(
    model_dir="models/",
    device="GPU",
)

results = detector.predict("data/YALE_Processed_Stage_6 (Done Alignment)/All/Ca Dao Ngạn Ngư Phương Ngôn, Nôm/images/")
