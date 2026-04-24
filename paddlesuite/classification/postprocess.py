# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import copy
import shutil
from functools import partial
import importlib
import numpy as np
import paddle.nn.functional as F
from .result import ImageClassificationResult


def build_postprocess(config):
    if config is None:
        return None

    mod = importlib.import_module(__name__)
    config = copy.deepcopy(config)

    main_indicator = config.pop(
        "main_indicator") if "main_indicator" in config else None
    main_indicator = main_indicator if main_indicator else ""

    func_list = []
    for func in config:
        func_list.append(getattr(mod, func)(**config[func]))
    return PostProcesser(func_list, main_indicator)


def parse_class_id_map(class_id_map_file, delimiter):
    if class_id_map_file is None:
        return None

    if not os.path.exists(class_id_map_file):
        print(
            "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
        )
        return None

    try:
        class_id_map = {}
        with open(class_id_map_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                partition = line.split("\n")[0].partition(delimiter)
                class_id_map[int(partition[0])] = str(partition[-1])
    except Exception as ex:
        print(ex)
        class_id_map = None
    return class_id_map


class PostProcesser(object):
    def __init__(self, func_list, main_indicator="Topk"):
        self.func_list = func_list
        self.main_indicator = main_indicator

    def __call__(self, x, image_file=None):
        rtn = None
        for func in self.func_list:
            tmp = func(x, image_file)
            if type(func).__name__ in self.main_indicator:
                rtn = tmp
        return rtn


class ScoreOutput(object):
    def __init__(self, decimal_places):
        self.decimal_places = decimal_places

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            score = np.around(x[idx], self.decimal_places).tolist()
            result = ImageClassificationResult(
                class_ids=[],
                scores=score,
                file_name=file_names[idx] if file_names is not None else None
            )
            y.append(result)
        return y


class Topk(object):
    def __init__(self,
                 topk=1,
                 class_id_map_file=None,
                 delimiter=None,
                 label_list=None):
        assert isinstance(topk, (int, ))
        self.topk = topk
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(
            class_id_map_file, delimiter) if not label_list else label_list

    def __call__(self, x, file_names=None):
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = ImageClassificationResult(
                class_ids=clas_id_list,
                scores=np.around(score_list, decimals=5).tolist(),
                file_name=file_names[idx] if file_names is not None else None,
                label_names=label_name_list if label_name_list else None
            )
            y.append(result)
        return y

class SavePreLabel(object):
    def __init__(self, save_dir):
        if save_dir is None:
            raise Exception(
                "Please specify save_dir if SavePreLabel specified.")
        self.save_dir = partial(os.path.join, save_dir)

    def __call__(self, x, file_names=None):
        if file_names is None:
            return
        assert x.shape[0] == len(file_names)
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-1].astype("int32")
            self.save(index, file_names[idx])

    def save(self, id, image_file):
        output_dir = self.save_dir(str(id))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(image_file, output_dir)
