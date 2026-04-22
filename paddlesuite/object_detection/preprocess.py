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

import cv2
import numpy as np


def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, "rb") as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype="uint8")
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info["im_shape"] = np.array(im.shape[:2], dtype=np.float32)
    im_info["scale_factor"] = np.array([1.0, 1.0], dtype=np.float32)
    return im, im_info


class MultiscaleTestResize(object):
    def __init__(
        self, origin_target_size=[800, 1333], target_size=[], interp=2, use_flip=True
    ):
        self.origin_target_size = origin_target_size
        self.target_size = target_size
        self.interp = interp
        self.use_flip = use_flip

    def __call__(self, im, im_info):
        ims, im_infos = [], []
        # 1. Origin size
        r = Resize(self.origin_target_size, keep_ratio=True, interp=self.interp)
        im_r, info_r = r(im.copy(), im_info.copy())
        ims.append(im_r)
        im_infos.append(info_r)

        # 2. Flipped origin size
        if self.use_flip:
            im_flip = im_r[:, ::-1, :]
            info_flip = info_r.copy()
            # Mark it so we can un-flip the boxes later
            info_flip["is_flipped"] = True
            ims.append(im_flip)
            im_infos.append(info_flip)

        # 3. Extra target sizes
        for size in self.target_size:
            r = Resize(size, keep_ratio=True, interp=self.interp)
            im_r, info_r = r(im.copy(), im_info.copy())
            ims.append(im_r)
            im_infos.append(info_r)

        return ims, im_infos


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp
        )
        im_info["im_shape"] = np.array(im.shape[:2]).astype("float32")
        im_info["scale_factor"] = np.array([im_scale_y, im_scale_x]).astype("float32")
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type="mean_std"):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == "mean_std":
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(
        self,
    ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


class PadStride(object):
    """padding image for model with FPN, instead PadBatch(pad_to_stride) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return im, im_info
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im, im_info


class Pad(object):
    def __init__(self, size, fill_value=[114.0, 114.0, 114.0]):
        """
        Pad image to a specified size.
        Args:
            size (list[int]): image target size
            fill_value (list[float]): rgb value of pad area, default (114.0, 114.0, 114.0)
        """
        super(Pad, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.fill_value = fill_value

    def __call__(self, im, im_info):
        im_h, im_w = im.shape[:2]
        h, w = self.size
        if h == im_h and w == im_w:
            im = im.astype(np.float32)
            return im, im_info

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
        im = canvas
        return im, im_info


def preprocess(im, preprocess_ops):
    im_info = {
        "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
        "im_shape": None,
        "is_flipped": False,
    }
    im, im_info = decode_image(im, im_info)

    # Handle branching for multi-scale
    ims, im_infos = [im], [im_info]
    for operator in preprocess_ops:
        next_ims, next_infos = [], []
        for i, info in zip(ims, im_infos):
            out_im, out_info = operator(i, info)
            if isinstance(out_im, list):
                next_ims.extend(out_im)
                next_infos.extend(out_info)
            else:
                next_ims.append(out_im)
                next_infos.append(out_info)
        ims, im_infos = next_ims, next_infos

    return ims, im_infos
