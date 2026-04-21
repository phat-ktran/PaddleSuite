from api import Detector

# 1. Initialize
detector = Detector(
    model_dir="models/faster_rcnn_r50_fpn_1x_hisdoc1b",
    device="GPU",
    batch_size=1,
    threshold=0.75
)

results = detector.predict("data/YALE_Processed_Stage_6 (Done Alignment)/All/Ca Dao Ngạn Ngư Phương Ngôn, Nôm/images/")

for result in results:
    result.save_to_json("output_jsons")
    result.save_to_img("output_images")
