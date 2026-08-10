import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from paddlesuite.object_detection.predictor import (
    ObjectDetector,
    create_inputs,
    nms,
)
from paddlesuite.object_detection.preprocess import NormalizeImage, Permute, Resize


def make_detector(workers):
    detector = ObjectDetector.__new__(ObjectDetector)
    detector.workers = workers
    detector._executor = (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    )
    detector.preprocess_ops = [
        Resize([48, 64], keep_ratio=False),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        Permute(),
    ]
    detector.pred_config = SimpleNamespace(labels=["char"])
    detector.threshold = 0.5
    return detector


class ObjectDetectorWorkersTest(unittest.TestCase):
    def test_workers_validation(self):
        for workers in (1, 4):
            ObjectDetector._validate_workers(workers)

        for workers in (True, 1.5, "2"):
            with self.assertRaises(TypeError):
                ObjectDetector._validate_workers(workers)

        for workers in (0, -1):
            with self.assertRaises(ValueError):
                ObjectDetector._validate_workers(workers)

    def test_create_inputs_pads_directly_into_batch(self):
        images = [
            np.ones((3, 2, 4), dtype=np.float32),
            np.full((3, 3, 2), 2, dtype=np.float32),
        ]
        infos = [
            {
                "im_shape": np.array([2, 4], dtype=np.float32),
                "scale_factor": np.array([1, 1], dtype=np.float32),
            },
            {
                "im_shape": np.array([3, 2], dtype=np.float32),
                "scale_factor": np.array([2, 2], dtype=np.float32),
            },
        ]

        inputs = create_inputs(images, infos)

        self.assertEqual(inputs["image"].shape, (2, 3, 3, 4))
        np.testing.assert_array_equal(inputs["image"][0, :, :2, :4], images[0])
        np.testing.assert_array_equal(inputs["image"][1, :, :3, :2], images[1])
        np.testing.assert_array_equal(inputs["image"][0, :, 2:, :], 0)
        np.testing.assert_array_equal(inputs["image"][1, :, :, 2:], 0)
        np.testing.assert_array_equal(
            inputs["im_shape"], np.array([[2, 4], [3, 2]], dtype=np.float32)
        )

    def test_nms_suppresses_overlaps_and_filters_scores(self):
        boxes = np.array(
            [
                [0, 0.90, 0, 0, 10, 10],
                [0, 0.80, 1, 1, 9, 9],
                [0, 0.70, 20, 20, 30, 30],
                [0, 0.01, 40, 40, 50, 50],
            ],
            dtype=np.float32,
        )

        result = nms(boxes, nms_threshold=0.5, score_threshold=0.01)

        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [0, 0.90, 0, 0, 10, 10],
                    [0, 0.70, 20, 20, 30, 30],
                ],
                dtype=np.float32,
            ),
        )

    def test_parallel_preprocess_matches_serial_and_preserves_order(self):
        rng = np.random.default_rng(123)
        images = [
            rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            for height, width in ((31, 42), (25, 39), (44, 29), (37, 51))
        ]
        serial = make_detector(workers=1)
        parallel = make_detector(workers=4)
        try:
            serial_inputs = serial._preprocess_batch(images)
            parallel_inputs = parallel._preprocess_batch(images)
        finally:
            parallel._executor.shutdown()

        self.assertEqual(len(serial_inputs), len(parallel_inputs))
        for serial_scale, parallel_scale in zip(serial_inputs, parallel_inputs):
            self.assertEqual(serial_scale.keys(), parallel_scale.keys())
            for key in ("image", "im_shape", "scale_factor"):
                np.testing.assert_allclose(serial_scale[key], parallel_scale[key])
            self.assertEqual(serial_scale["is_flipped"], parallel_scale["is_flipped"])

    def test_parallel_postprocess_matches_serial_and_preserves_order(self):
        image_boxes = [
            [
                np.array(
                    [
                        [0, 0.9, index, 0, index + 10, 10],
                        [0, 0.8, index + 1, 1, index + 9, 9],
                        [0, 0.7, index, 20, index + 5, 25],
                    ],
                    dtype=np.float32,
                )
            ]
            for index in range(4)
        ]
        items = list(zip([f"image-{index}.jpg" for index in range(4)], image_boxes))
        serial = make_detector(workers=1)
        parallel = make_detector(workers=4)
        try:
            serial_results = serial._parallel_map(serial._postprocess_image, items)
            parallel_results = parallel._parallel_map(
                parallel._postprocess_image, items
            )
        finally:
            parallel._executor.shutdown()

        self.assertEqual(
            [result.file_name for result in serial_results],
            [result.file_name for result in parallel_results],
        )
        for serial_result, parallel_result in zip(serial_results, parallel_results):
            np.testing.assert_array_equal(serial_result.cls_id, parallel_result.cls_id)
            np.testing.assert_allclose(serial_result.score, parallel_result.score)
            np.testing.assert_allclose(serial_result.boxes, parallel_result.boxes)


if __name__ == "__main__":
    unittest.main()
