# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import yaml
import numpy as np
import cv2
import paddle
from PIL import Image
from paddle.inference import Config
from paddle.inference import create_predictor

from paddlesuite.base import BasePredictor
from .preprocess import create_operators
from .postprocess import build_postprocess


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


class PredictConfig:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, "infer_cfg.yaml")
        if not os.path.exists(deploy_file):
            raise FileNotFoundError(f"Config file not found: {deploy_file}")

        with open(deploy_file, "r") as f:
            self.yml_conf = yaml.safe_load(f)

        if not self.yml_conf:
            raise ValueError(f"Empty or invalid config file: {deploy_file}")

        self.global_conf = self.yml_conf.get("Global", {})
        self.preprocess_infos = self.yml_conf.get("PreProcess", {}).get(
            "transform_ops", []
        )
        self.postprocess_conf = self.yml_conf.get("PostProcess", {})


def load_predictor(
    model_dir,
    device="CPU",
    cpu_threads=1,
    enable_mkldnn=False,
):
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    if not os.path.exists(model_file) or not os.path.exists(params_file):
        # Try checking for PIR mode or different naming if necessary
        # But per instructions, we follow the simplified structure
        pass

    config = Config(model_file, params_file)

    if device == "GPU":
        config.enable_use_gpu(200, 0)
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
            except Exception:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
        config.set_cpu_math_library_num_threads(cpu_threads)

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


class ImageClassifier(BasePredictor):
    def __init__(
        self,
        model_dir,
        device="CPU",
        cpu_threads=1,
        enable_mkldnn=False,
        batch_size=None,
    ):
        self.pred_config = PredictConfig(model_dir)
        if batch_size is not None:
            self.pred_config.global_conf["batch_size"] = batch_size
        self.preprocess_ops = create_operators(self.pred_config.preprocess_infos)
        self.postprocess = build_postprocess(self.pred_config.postprocess_conf)

        self.predictor, self.config = load_predictor(
            model_dir,
            device=device,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
        )

    def predict(self, data):
        if isinstance(data, (str, bytes)):
            image_list = get_files(data) if isinstance(data, str) else [data]
        elif isinstance(data, list):
            image_list = data
        else:
            return []

        if not image_list:
            return []

        return self._predict_images(image_list)

    def predict_iter(self, data):
        if isinstance(data, (str, bytes)):
            image_list = get_files(data) if isinstance(data, str) else [data]
        elif isinstance(data, list):
            image_list = data
        else:
            return

        if not image_list:
            return

        batch_size = self.pred_config.global_conf.get("batch_size", 1)
        for idx in range(0, len(image_list), batch_size):
            batch_list = image_list[idx : idx + batch_size]
            results = self._predict_images(batch_list)
            if results:
                for res in results:
                    yield res

    def _predict_images(self, image_list):
        results = []
        batch_size = self.pred_config.global_conf.get("batch_size", 1)

        for idx in range(0, len(image_list), batch_size):
            batch_list = image_list[idx : idx + batch_size]
            batch_data = []

            for im in batch_list:
                # Preprocess
                data = im
                if isinstance(im, str):
                    with open(im, "rb") as f:
                        data = f.read()
                elif isinstance(im, Image.Image):
                    # DecodeImage normally converts byte input to an RGB NumPy
                    # array.  Do the equivalent conversion for in-memory
                    # Pillow images so every subsequent operator receives the
                    # same representation as it does for file paths.
                    data = np.asarray(im.convert("RGB"))

                for ops in self.preprocess_ops:
                    data = ops(data)
                batch_data.append(data)

            # Inference
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])
            input_tensor.copy_from_cpu(np.array(batch_data))

            self.predictor.run()

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            batch_output = output_tensor.copy_to_cpu()

            # Postprocess
            if self.postprocess is not None:
                # We pass the image path if available for metadata
                image_file = [im if isinstance(im, str) else None for im in batch_list]
                batch_output = self.postprocess(batch_output, image_file)

            if batch_output:
                results.extend(batch_output)
            else:
                results.extend([None] * len(batch_list))

        return results
