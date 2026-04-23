import os
import json
import numpy as np


class ObjectDetectorResult:
    def __init__(self, file_name, boxes, labels, threshold):
        self.file_name = file_name
        self.labels = labels
        self.threshold = threshold

        if boxes is not None and len(boxes) > 0:
            self.cls_id = boxes[:, 0].astype(int)
            self.score = boxes[:, 1]
            self.boxes = boxes[:, 2:]
        else:
            self.cls_id = np.array([])
            self.score = np.array([])
            self.boxes = np.empty((0, 4))

    def save_to_json(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        img_name = os.path.basename(self.file_name)
        json_name = os.path.splitext(img_name)[0] + ".json"
        out_path = os.path.join(dir_path, json_name)

        bbox_results = []
        if len(self.boxes) > 0:
            for i in range(len(self.boxes)):
                # filter by threshold
                if self.score[i] < self.threshold:
                    continue
                box = self.boxes[i]
                bbox_results.append(
                    {
                        "image_id": json_name,
                        "category_id": int(self.cls_id[i]),
                        "file_name": img_name,
                        # xyxy -> xywh
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "score": float(self.score[i]),
                    }
                )

        with open(out_path, "w") as f:
            json.dump(bbox_results, f, indent=4)
        print(f"Saved JSON result to: {out_path}")

    def save_to_img(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        img_name = os.path.basename(self.file_name)
        out_path = os.path.join(dir_path, img_name)

        im_results = {}
        if len(self.boxes) > 0:
            im_results["boxes"] = np.column_stack((self.cls_id, self.score, self.boxes))

        from .visualize import visualize_box_mask

        im = visualize_box_mask(
            self.file_name, im_results, self.labels, threshold=self.threshold
        )

        im.save(out_path, quality=95)
        print("Saved Image result to: " + out_path)
