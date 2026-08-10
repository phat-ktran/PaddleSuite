import os
import yaml
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
)
from paddlesuite.base import BasePredictor
from .utils import Timer
from .result import ObjectDetectorResult


PREPROCESS_OPERATORS = {
    operator.__name__: operator
    for operator in (
        MultiscaleTestResize,
        Resize,
        NormalizeImage,
        Permute,
        PadStride,
        Pad,
    )
}


def nms(bboxes, nms_threshold=0.5, score_threshold=0.01):
    if len(bboxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = bboxes[bboxes[:, 1] > score_threshold]
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


def create_inputs(imgs, im_info):
    inputs = {}
    if len(imgs) == 1:
        inputs["image"] = np.expand_dims(np.asarray(imgs[0], dtype=np.float32), axis=0)
        inputs["im_shape"] = np.expand_dims(
            np.asarray(im_info[0]["im_shape"], dtype=np.float32), axis=0
        )
        inputs["scale_factor"] = np.expand_dims(
            np.asarray(im_info[0]["scale_factor"], dtype=np.float32), axis=0
        )
        return inputs

    inputs["im_shape"] = np.asarray([e["im_shape"] for e in im_info], dtype=np.float32)
    inputs["scale_factor"] = np.asarray(
        [e["scale_factor"] for e in im_info], dtype=np.float32
    )

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    batch = np.zeros(
        (len(imgs), imgs[0].shape[0], max_shape_h, max_shape_w),
        dtype=np.float32,
    )
    for image_idx, img in enumerate(imgs):
        im_c, im_h, im_w = img.shape[:]
        batch[image_idx, :im_c, :im_h, :im_w] = img
    inputs["image"] = batch
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
            except Exception:
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
        workers=1,
    ):
        self._validate_workers(workers)
        self.workers = workers
        self.pred_config = self.set_config(model_dir, use_fd_format=use_fd_format)
        self.preprocess_ops = self._create_preprocess_ops()
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=1,
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
        self._executor = (
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="paddlesuite-object-detector",
            )
            if workers > 1
            else None
        )

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig(model_dir, use_fd_format=use_fd_format)

    def _create_preprocess_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop("type")
            operator = PREPROCESS_OPERATORS.get(op_type)
            if operator is None:
                raise ValueError(f"Unsupported preprocess operator: {op_type}")
            preprocess_ops.append(operator(**new_op_info))
        return preprocess_ops

    def _parallel_map(self, function, items):
        if self._executor is None or len(items) < 2:
            return [function(item) for item in items]
        return list(self._executor.map(function, items))

    def _run_preprocess(self, image):
        return preprocess(image, self.preprocess_ops)

    def _preprocess_one(self, image):
        im_scales, info_scales = self._run_preprocess(image)
        if not isinstance(im_scales, list):
            im_scales = [im_scales]
            info_scales = [info_scales]

        inputs_list = []
        for im, im_info in zip(im_scales, info_scales):
            inputs = create_inputs([im], [im_info])
            inputs["is_flipped"] = im_info.get("is_flipped", False)
            inputs_list.append(inputs)

        return inputs_list

    def preprocess(self, image_list):
        per_image_inputs = self._parallel_map(self._preprocess_one, image_list)
        return [inputs for image_inputs in per_image_inputs for inputs in image_inputs]

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

    def predict(self, data, batch_size=1):
        self._validate_batch_size(batch_size)
        if isinstance(data, str):
            image_list = get_files(data)
        else:
            image_list = data
        if not image_list:
            return []
        return self._predict_images(image_list, batch_size=batch_size)

    def predict_iter(self, data, batch_size=1):
        self._validate_batch_size(batch_size)
        if isinstance(data, str):
            image_list = get_files(data)
        else:
            image_list = data
        if not image_list:
            return

        for idx in range(0, len(image_list), batch_size):
            results = self._predict_images(
                image_list[idx : idx + batch_size], batch_size=batch_size
            )
            for res in results:
                yield res

    @staticmethod
    def _validate_batch_size(batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("`batch_size` must be an integer")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be greater than 0")

    @staticmethod
    def _validate_workers(workers):
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise TypeError("`workers` must be an integer")
        if workers <= 0:
            raise ValueError("`workers` must be greater than 0")

    def _preprocess_batch(self, image_list):
        per_image_scales = self._parallel_map(self._run_preprocess, image_list)
        scale_counts = {len(images) for images, _ in per_image_scales}
        if len(scale_counts) != 1:
            raise ValueError(
                "All images in a batch must have the same number of scales"
            )

        inputs_list = []
        for scale_idx in range(scale_counts.pop()):
            images = [images[scale_idx] for images, _ in per_image_scales]
            im_info = [infos[scale_idx] for _, infos in per_image_scales]
            batch_inputs = create_inputs(images, im_info)
            batch_inputs["is_flipped"] = [
                info.get("is_flipped", False) for info in im_info
            ]
            inputs_list.append(batch_inputs)

        return inputs_list

    def _postprocess_image(self, item):
        im_path, image_boxes = item
        if image_boxes:
            concatenated_boxes = np.concatenate(image_boxes)
            merged_boxes = nms(
                concatenated_boxes,
                nms_threshold=0.5,
            )
        else:
            merged_boxes = np.empty((0, 6), dtype=np.float32)

        return ObjectDetectorResult(
            file_name=im_path,
            boxes=merged_boxes,
            labels=self.pred_config.labels,
            threshold=self.threshold,
        )

    def _predict_images(self, image_list, batch_size=1):
        self._validate_batch_size(batch_size)
        results = []
        for idx in range(0, len(image_list), batch_size):
            batch_list = image_list[idx : idx + batch_size]
            self.det_times.preprocess_time_s.start()
            inputs_list = self._preprocess_batch(batch_list)
            self.det_times.preprocess_time_s.end()

            all_boxes = [[] for _ in batch_list]

            for inputs in inputs_list:
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
                boxes = scale_result.get("boxes")
                boxes_num = np.asarray(scale_result["boxes_num"]).reshape(-1)
                if len(boxes_num) != len(batch_list):
                    raise RuntimeError(
                        "The detector did not return `boxes_num` for every image "
                        "in the batch"
                    )

                box_offset = 0
                for image_idx, box_count in enumerate(boxes_num.astype(int)):
                    np_boxes = boxes[box_offset : box_offset + box_count]
                    box_offset += box_count
                    if len(np_boxes) == 0:
                        continue

                    if inputs["is_flipped"][image_idx]:
                        orig_w = (
                            inputs["im_shape"][image_idx][1]
                            / inputs["scale_factor"][image_idx][1]
                        )
                        xmin = orig_w - np_boxes[:, 4]
                        xmax = orig_w - np_boxes[:, 2]
                        np_boxes[:, 2] = xmin
                        np_boxes[:, 4] = xmax

                    all_boxes[image_idx].append(np_boxes)

            self.det_times.postprocess_time_s.start()
            results.extend(
                self._parallel_map(
                    self._postprocess_image,
                    list(zip(batch_list, all_boxes)),
                )
            )

            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += len(batch_list)

        return results
