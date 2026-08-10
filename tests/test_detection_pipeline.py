import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paddlesuite.examples.run import (
    BookOutput,
    postprocess_batch_task,
    queued_batch_count,
    sort_box_indices,
)


class DetectionPipelineTest(unittest.TestCase):
    def test_book_output_buffers_out_of_order_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            book_path = Path(directory) / "book"
            (book_path / "images").mkdir(parents=True)
            output = BookOutput(str(book_path))
            self.assertEqual(output.register_batch(), 0)
            self.assertEqual(output.register_batch(), 1)
            output.seal()

            output.accept_batch(1, True, ["second\n"])
            self.assertEqual(queued_batch_count({}, {str(book_path): output}), 1)
            self.assertEqual((book_path / "images" / "char_det.txt").read_text(), "")

            output.accept_batch(0, True, ["first\n"])
            result = output.finish()

            self.assertTrue(result[1])
            self.assertEqual(
                (book_path / "images" / "char_det.txt").read_text(),
                "first\nsecond\n",
            )

    def test_sort_box_indices_preserves_polygon_score_association(self):
        polygons = [
            [[80, 20], [90, 20], [90, 30], [80, 30]],
            [[80, 0], [90, 0], [90, 10], [80, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ]

        self.assertEqual(sort_box_indices(polygons), [1, 0, 2])

    def test_postprocess_batch_task_returns_ordered_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            book_path = Path(directory) / "book"
            (book_path / "images").mkdir(parents=True)
            boxes = np.array(
                [
                    [0, 0.90, 80, 20, 90, 30],
                    [0, 0.80, 80, 0, 90, 10],
                    [0, 0.70, 20, 0, 30, 10],
                    [0, 0.40, 40, 0, 50, 10],
                ],
                dtype=np.float32,
            )
            raw_results = [(str(book_path / "images" / "page.jpg"), [boxes])]

            result = postprocess_batch_task((str(book_path), 3, raw_results, 0.5))

            self.assertEqual(result[0], str(book_path))
            self.assertEqual(result[1], 3)
            self.assertTrue(result[2])
            line = result[3][0]
            image_name, payload = line.rstrip("\n").split("\t")
            detections = json.loads(payload)
            self.assertEqual(image_name, "images/page.jpg")
            np.testing.assert_allclose(
                [detection["dt_score"] for detection in detections],
                [0.8, 0.9, 0.7],
            )


if __name__ == "__main__":
    unittest.main()
