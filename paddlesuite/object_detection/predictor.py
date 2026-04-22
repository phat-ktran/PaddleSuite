import os
import yaml
import numpy as np
import math
import json
import paddle
from dataclasses import dataclass


from paddle.inference import Config
from paddle.inference import create_predictor

from .preprocess import (
    preprocess,
    MultiscaleTestResize,
    Resize,
    NormalizeImage,
    Permute,
    PadStride,
    Pad,
    decode_image,
)
from .utils import Timer
from .visualize import visualize_box_mask


@dataclass
class ObjectDetectionResult:
    file_name: str
    boxes: np.ndarray
    masks: np.ndarray
    labels: list
    threshold: float

    def save_to_json(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        img_name = os.path.basename(self.file_name)
        json_name = os.path.splitext(img_name)[0] + ".json"
        out_path = os.path.join(dir_path, json_name)

        bbox_results = []
        if self.boxes is not None and len(self.boxes) > 0:
            for box in self.boxes:
                # filter by threshold
                if box[1] < self.threshold:
                    continue
                bbox_results.append(
                    {
                        "image_id": json_name,
                        "category_id": int(box[0]),
                        "file_name": img_name,
                        # xyxy -> xywh
                        "bbox": [
                            float(box[2]),
                            float(box[3]),
                            float(box[4] - box[2]),
                            float(box[5] - box[3]),
                        ],
                        "score": float(box[1]),
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
        if self.boxes is not None:
            im_results["boxes"] = self.boxes
        if self.masks is not None:
            im_results["masks"] = self.masks

        im = visualize_box_mask(
            self.file_name, im_results, self.labels, threshold=self.threshold
        )

        im.save(out_path, quality=95)
        print("Saved Image result to: " + out_path)


def compute_iou(box1, boxes):
    x1 = np.maximum(box1[:, 0], boxes[:, 0])
    y1 = np.maximum(box1[:, 1], boxes[:, 1])
    x2 = np.minimum(box1[:, 2], boxes[:, 2])
    y2 = np.minimum(box1[:, 3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return intersection / (area1 + area2 - intersection)


def multiclass_nms(bboxes, num_classes=80, nms_threshold=0.5, score_threshold=0.01):
    if len(bboxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    keep_boxes = []
    for class_id in range(num_classes):
        class_indices = np.where(bboxes[:, 0] == class_id)[0]
        if len(class_indices) == 0:
            continue

        class_boxes = bboxes[class_indices]
        class_scores = class_boxes[:, 1]

        score_mask = class_scores > score_threshold
        class_boxes = class_boxes[score_mask]

        if len(class_boxes) == 0:
            continue

        order = class_boxes[:, 1].argsort()[::-1]
        class_boxes = class_boxes[order]

        keep = []
        while class_boxes.shape[0] > 0:
            keep.append(class_boxes[0])
            if class_boxes.shape[0] == 1:
                break

            ious = compute_iou(class_boxes[0:1, 2:], class_boxes[1:, 2:])
            class_boxes = class_boxes[1:][ious < nms_threshold]

        keep_boxes.extend(keep)

    if len(keep_boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(keep_boxes)


def create_inputs(imgs, im_info):
    inputs = {}
    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs["image"] = np.array((imgs[0],)).astype("float32")
        inputs["im_shape"] = np.array((im_info[0]["im_shape"],)).astype("float32")
        inputs["scale_factor"] = np.array((im_info[0]["scale_factor"],)).astype(
            "float32"
        )
        return inputs

    for e in im_info:
        im_shape.append(np.array((e["im_shape"],)).astype("float32"))
        scale_factor.append(np.array((e["scale_factor"],)).astype("float32"))

    inputs["im_shape"] = np.concatenate(im_shape, axis=0)
    inputs["scale_factor"] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs["image"] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig:
    def __init__(self, model_dir, use_fd_format=False):
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        with open(deploy_file, "r") as f:
            yml_conf = yaml.safe_load(f)

        self.check_model(yml_conf)
        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]
        self.min_subgraph_size = yml_conf["min_subgraph_size"]
        self.labels = yml_conf["label_list"]
        self.mask = False
        self.use_dynamic_shape = yml_conf["use_dynamic_shape"]
        if "mask" in yml_conf:
            self.mask = yml_conf["mask"]
        self.tracker = None
        if "tracker" in yml_conf:
            self.tracker = yml_conf["tracker"]
        if "NMS" in yml_conf:
            self.nms = yml_conf["NMS"]
        if "fpn_stride" in yml_conf:
            self.fpn_stride = yml_conf["fpn_stride"]

    def check_model(self, yml_conf):
        for op_info in yml_conf["Preprocess"]:
            op_type = op_info["type"]
            if op_type == "NormalizeImage":
                op_info["is_scale"] = op_info.get("is_scale", True)
                op_info["mean"] = op_info.get("mean", [0.485, 0.456, 0.406])
                op_info["std"] = op_info.get("std", [0.229, 0.224, 0.225])


def load_predictor(
    model_dir,
    arch,
    run_mode="paddle",
    batch_size=1,
    min_subgraph_size=3,
    device="CPU",
    use_dynamic_shape=False,
    trt_min_shape=1,
    trt_max_shape=1280,
    trt_opt_shape=640,
    trt_calib_mode=False,
    cpu_threads=1,
    enable_mkldnn=False,
    enable_mkldnn_bfloat16=False,
    delete_shuffle_pass=False,
):
    if device != "GPU" and run_mode != "paddle":
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(
                run_mode, device
            )
        )
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams"),
    )
    if device == "GPU":
        config.enable_use_gpu(200, 0)
        if run_mode != "paddle":
            precision_map = {
                "trt_int8": Config.Precision.Int8,
                "trt_fp32": Config.Precision.Float32,
                "trt_fp16": Config.Precision.Half,
            }
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[run_mode],
                use_static=False,
                use_calib_mode=trt_calib_mode,
            )
            if use_dynamic_shape:
                min_input_shape = {
                    "image": [batch_size, 3, trt_min_shape, trt_min_shape],
                    "scale_factor": [batch_size, 2],
                }
                max_input_shape = {
                    "image": [batch_size, 3, trt_max_shape, trt_max_shape],
                    "scale_factor": [batch_size, 2],
                }
                opt_input_shape = {
                    "image": [batch_size, 3, trt_opt_shape, trt_opt_shape],
                    "scale_factor": [batch_size, 2],
                }
                config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape
                )
    elif device == "XPU":
        config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == "NPU":
        config.enable_lite_engine()
        config.enable_npu()
    else:
        config.disable_gpu()
        if enable_mkldnn:
            try:
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
        config.set_cpu_math_library_num_threads(cpu_threads)

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


def get_files(path):
    if os.path.isfile(path):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            return [path]
        return None

    if os.path.isdir(path):
        image_files = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(path, file))
        return sorted(image_files)

    return None


from paddlesuite.base import BasePredictor

class ObjectDetector(BasePredictor):
    def __init__(
        self,
        model_dir,
        device="CPU",
        run_mode="paddle",
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        enable_mkldnn_bfloat16=False,
        threshold=0.5,
        delete_shuffle_pass=False,
        use_fd_format=False,
    ):
        self.pred_config = self.set_config(model_dir, use_fd_format=use_fd_format)
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass,
        )
        self.det_times = Timer()
        self.threshold = threshold
        self.device = device

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig(model_dir, use_fd_format=use_fd_format)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop("type")
            preprocess_ops.append(eval(op_type)(**new_op_info))

        inputs_list = []
        for im_path in image_list:
            im_scales, info_scales = preprocess(im_path, preprocess_ops)

            if not isinstance(im_scales, list):
                im_scales = [im_scales]
                info_scales = [info_scales]

            for im, im_info in zip(im_scales, info_scales):
                inputs = create_inputs([im], [im_info])
                inputs_list.append(inputs)

        return inputs_list

    def postprocess(self, inputs, result):
        np_boxes_num = result["boxes_num"]
        assert isinstance(
            np_boxes_num, np.ndarray
        ), "`np_boxes_num` should be a `numpy.ndarray`"
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def _predict(self):
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        if len(output_names) == 1:
            np_boxes_num = np.array([len(np_boxes)])
        else:
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()

        np_masks = None
        if self.pred_config.mask:
            masks_tensor = self.predictor.get_output_handle(output_names[2])
            np_masks = masks_tensor.copy_to_cpu()

        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def predict(self, data):
        if isinstance(data, str):
            image_list = get_files(data)
        else:
            image_list = data
        if not image_list:
            return []
        return self._predict_images(image_list)

    def predict_iter(self, data):
        if isinstance(data, str):
            image_list = get_files(data)
        else:
            image_list = data
        if not image_list:
            return

        for im_path in image_list:
            results = self._predict_images([im_path])
            for res in results:
                yield res

    def _predict_images(self, image_list):
        results = []
        for im_path in image_list:
            self.det_times.preprocess_time_s.start()
            inputs_list = self.preprocess([im_path])
            if not isinstance(inputs_list, list):
                inputs_list = [inputs_list]
            self.det_times.preprocess_time_s.end()

            all_boxes = []

            for scale_idx, inputs in enumerate(inputs_list):
                self.det_times.inference_time_s.start()
                input_names = self.predictor.get_input_names()
                for j in range(len(input_names)):
                    input_tensor = self.predictor.get_input_handle(input_names[j])
                    if input_names[j] == "x":
                        input_tensor.copy_from_cpu(inputs["image"])
                    else:
                        input_tensor.copy_from_cpu(inputs[input_names[j]])

                result = self._predict()
                self.det_times.inference_time_s.end()

                scale_result = self.postprocess(inputs, result)
                if (
                    scale_result.get("boxes") is not None
                    and len(scale_result["boxes"]) > 0
                ):
                    np_boxes = scale_result["boxes"]
                    if inputs.get("is_flipped", False):
                        orig_w = (
                            inputs["im_shape"][0][1] / inputs["scale_factor"][0][1]
                        )
                        xmin = orig_w - np_boxes[:, 4]
                        xmax = orig_w - np_boxes[:, 2]
                        np_boxes[:, 2] = xmin
                        np_boxes[:, 4] = xmax

                    all_boxes.append(np_boxes)

            self.det_times.postprocess_time_s.start()
            if len(all_boxes) > 0:
                concatenated_boxes = np.concatenate(all_boxes)
                # Pass the number of classes from labels
                merged_boxes = multiclass_nms(
                    concatenated_boxes,
                    num_classes=len(self.pred_config.labels),
                    nms_threshold=0.5,
                )
            else:
                merged_boxes = np.empty((0, 6), dtype=np.float32)

            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1

            # Create the ObjectDetectionResult
            det_result = ObjectDetectionResult(
                file_name=im_path,
                boxes=merged_boxes,
                masks=None,  # Masks logic can be added later if needed
                labels=self.pred_config.labels,
                threshold=self.threshold,
            )

            results.append(det_result)

        return results
