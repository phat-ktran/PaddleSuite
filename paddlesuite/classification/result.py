# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

class ImageClassificationResult:
    def __init__(self, class_ids, scores, file_name=None, label_names=None):
        self.class_ids = class_ids.tolist() if isinstance(class_ids, np.ndarray) else class_ids
        self.scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        self.file_name = file_name
        self.label_names = label_names

    def __str__(self):
        res = ""
        if self.file_name:
            res += f"{self.file_name}:\t"
        
        scores_str = "[{}]".format(", ".join("{:.2f}".format(r) for r in self.scores))
        
        if self.label_names:
            res += f"class id(s): {self.class_ids}, score(s): {scores_str}, label_name(s): {self.label_names}"
        else:
            res += f"class id(s): {self.class_ids}, score(s): {scores_str}"
        return res

    def __repr__(self):
        return f"ImageClassificationResult(file_name={self.file_name}, class_ids={self.class_ids}, scores={self.scores}, label_names={self.label_names})"
