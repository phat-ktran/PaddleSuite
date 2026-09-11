import argparse
import json
import logging
import multiprocessing as mp
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from paddlesuite.object_detection.postprocess import merge_scale_boxes

# Allow large images to be processed by Pillow.
Image.MAX_IMAGE_PIXELS = None


def configure_logging():
    logging.basicConfig(
        filename="detection_pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def sort_box_indices(boxes):
    """Return RTL, top-to-bottom reading-order indices for OCR polygons."""
    if boxes is None or len(boxes) == 0:
        return []

    polygons = np.asarray(boxes, dtype=np.float64)
    centers = polygons.mean(axis=1)
    widths = np.ptp(polygons[:, :, 0], axis=1)
    edges = polygons[:, 1] - polygons[:, 0]
    global_angle = np.median(np.arctan2(edges[:, 1], edges[:, 0]))

    cos_angle = np.cos(global_angle)
    sin_angle = np.sin(global_angle)
    rotated_x = centers[:, 0] * cos_angle + centers[:, 1] * sin_angle
    rotated_y = -centers[:, 0] * sin_angle + centers[:, 1] * cos_angle

    x_tolerance = widths.mean() * 0.6
    right_to_left = np.argsort(-rotated_x, kind="stable")

    columns = []
    current_column = []
    current_x_sum = 0.0
    for index in right_to_left:
        index = int(index)
        if (
            current_column
            and abs(rotated_x[index] - current_x_sum / len(current_column))
            > x_tolerance
        ):
            columns.append(current_column)
            current_column = []
            current_x_sum = 0.0

        current_column.append(index)
        current_x_sum += rotated_x[index]

    if current_column:
        columns.append(current_column)

    ordered_indices = []
    for column in columns:
        column.sort(key=lambda index: rotated_y[index])
        ordered_indices.extend(column)
    return ordered_indices


def sort_box(boxes):
    """Sort OCR polygons in right-to-left, top-to-bottom reading order."""
    return [boxes[index] for index in sort_box_indices(boxes)]


def normalize_to_poly(box):
    """Convert ``[xmin, ymin, xmax, ymax]`` to a four-point polygon."""
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def load_book_images(book_path_str):
    """Return a book's sorted JPG paths or an error message."""
    images_dir = Path(book_path_str) / "images"
    if not images_dir.exists():
        return None, "No 'images' subdirectory found."

    image_paths = sorted(str(path) for path in images_dir.glob("*.jpg"))
    if not image_paths:
        return None, "No JPG images found in directory."
    return image_paths, None


def postprocess_batch_task(task_args):
    """Run CPU NMS, reading-order sorting, and JSON encoding for one batch."""
    book_path_str, batch_index, raw_results, score_threshold = task_args
    try:
        lines = []
        for image_path, image_boxes in raw_results:
            image_filename = Path(image_path).name
            merged_boxes = merge_scale_boxes(
                image_boxes,
                nms_threshold=0.5,
                score_threshold=score_threshold,
            )

            if len(merged_boxes) == 0:
                lines.append(f"images/{image_filename}\t[]\n")
                continue

            polygons = [normalize_to_poly(box) for box in merged_boxes[:, 2:6]]
            ordered_indices = sort_box_indices(polygons)
            json_objects = [
                {
                    "points": np.asarray(polygons[index]).tolist(),
                    "dt_score": float(merged_boxes[index, 1]),
                }
                for index in ordered_indices
            ]
            json_str = json.dumps(json_objects, ensure_ascii=False)
            lines.append(f"images/{image_filename}\t{json_str}\n")

        return book_path_str, batch_index, True, lines
    except Exception as error:
        return book_path_str, batch_index, False, str(error)


class BookOutput:
    """Write asynchronously processed batches in their original image order."""

    def __init__(self, book_path):
        self.book_path = book_path
        output_path = Path(book_path) / "images" / "char_det.txt"
        self.output_file = open(output_path, "w", encoding="utf-8")
        self.submitted_batches = 0
        self.completed_batches = 0
        self.completed_images = 0
        self.next_batch = 0
        self.buffered_lines = {}
        self.error = None
        self.sealed = False

    def register_batch(self):
        batch_index = self.submitted_batches
        self.submitted_batches += 1
        return batch_index

    def accept_batch(self, batch_index, success, payload):
        self.completed_batches += 1
        if not success:
            if self.error is None:
                self.error = str(payload)
            self.buffered_lines.clear()
            return
        if self.error is not None:
            return

        self.buffered_lines[batch_index] = payload
        while self.next_batch in self.buffered_lines:
            lines = self.buffered_lines.pop(self.next_batch)
            self.output_file.writelines(lines)
            self.completed_images += len(lines)
            self.next_batch += 1

    def seal(self, error=None):
        self.sealed = True
        if error is not None and self.error is None:
            self.error = str(error)
            self.buffered_lines.clear()

    @property
    def ready(self):
        return self.sealed and self.completed_batches == self.submitted_batches

    def finish(self):
        self.output_file.close()
        if self.error is not None:
            return self.book_path, False, self.error
        return (
            self.book_path,
            True,
            f"Processed {self.completed_images} images.",
        )

    def close(self):
        if not self.output_file.closed:
            self.output_file.close()


def initialize_postprocess_worker():
    """Let the parent process exclusively handle interactive cancellation."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def shutdown_executor(executor, futures, terminate=False):
    """Stop pending work and optionally terminate running worker processes."""
    for future in futures:
        future.cancel()

    if not terminate:
        executor.shutdown(wait=True, cancel_futures=True)
        return

    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        terminate_workers()
        return

    # Python < 3.14 has no public immediate-termination API.
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + 5
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))

    for process in processes:
        if process.is_alive():
            process.kill()

    executor.shutdown(wait=True, cancel_futures=True)


def record_book_result(result, progress):
    """Log one completed book and return success/failure counts."""
    book_path, success, message = result
    if success:
        logging.info("Success [%s]: %s", Path(book_path).name, message)
    else:
        logging.error("Failed [%s]: %s", Path(book_path).name, message)
    progress.update(1)
    return (1, 0) if success else (0, 1)


def finish_ready_books(book_outputs, progress):
    success_count = 0
    fail_count = 0
    ready_books = [
        book_path for book_path, output in book_outputs.items() if output.ready
    ]
    for book_path in ready_books:
        result = book_outputs.pop(book_path).finish()
        succeeded, failed = record_book_result(result, progress)
        success_count += succeeded
        fail_count += failed
    return success_count, fail_count


def drain_postprocess_tasks(pending, book_outputs, progress, block):
    """Collect completed tasks, optionally waiting for at least one completion."""
    if not pending:
        return finish_ready_books(book_outputs, progress)

    if block:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
    else:
        completed = {future for future in pending if future.done()}

    for future in completed:
        book_path, batch_index = pending.pop(future)
        try:
            _, _, success, payload = future.result()
        except Exception as error:
            success, payload = False, str(error)
        book_outputs[book_path].accept_batch(batch_index, success, payload)
    return finish_ready_books(book_outputs, progress)


def queued_batch_count(pending, book_outputs):
    """Count raw/in-progress and processed-but-not-yet-writable batches."""
    buffered_batches = sum(
        len(output.buffered_lines) for output in book_outputs.values()
    )
    return len(pending) + buffered_batches


def discover_books(input_dir):
    """Find book directories under the input directory's chunk directories."""
    books = []
    chunk_dirs = sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for chunk_dir in chunk_dirs:
        books.extend(
            str(path)
            for path in sorted(
                (path for path in chunk_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
        )
    return books


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipelined batch character detection on chunked datasets."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=4,
        help="Number of model-free CPU postprocessing processes.",
    )
    parser.add_argument(
        "--predictor-workers",
        type=positive_integer,
        default=16,
        help="ObjectDetector preprocessing worker threads.",
    )
    parser.add_argument("--cpu-threads", type=positive_integer, default=1)
    parser.add_argument("--batch-size", type=positive_integer, default=20)
    parser.add_argument("--threshold", type=probability, default=0.88)
    parser.add_argument(
        "--max-pending-batches",
        type=positive_integer,
        default=None,
        help="Maximum batches not yet written to labels (default: workers).",
    )
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    print("Scanning chunks for book directories...")
    book_directories = discover_books(input_dir)
    total_books = len(book_directories)
    if total_books == 0:
        print("No valid book directories found inside the chunks.")
        return

    max_pending_batches = args.max_pending_batches or args.workers
    print(
        f"Found {total_books} books. Using one {args.device.upper()} predictor, "
        f"{args.workers} CPU postprocessing workers, and a "
        f"{max_pending_batches}-batch pending limit."
    )

    # Import Paddle only in the model-owning parent process. Spawned CPU workers
    # import the lightweight postprocess module without loading Paddle or a model.
    from paddlesuite.object_detection import ObjectDetector

    postprocess_executor = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_postprocess_worker,
        mp_context=mp.get_context("spawn"),
    )
    detector = None
    pending = {}
    book_outputs = {}
    success_count = 0
    fail_count = 0
    executor_shutdown = False

    try:
        detector = ObjectDetector(
            model_dir=args.model_dir,
            device=args.device.upper(),
            cpu_threads=args.cpu_threads,
            threshold=args.threshold,
            workers=args.predictor_workers,
        )

        with tqdm(total=total_books, desc="Processing Books") as progress:
            for book_path in book_directories:
                image_paths, error = load_book_images(book_path)
                if error is not None:
                    logging.error("Failed [%s]: %s", Path(book_path).name, error)
                    fail_count += 1
                    progress.update(1)
                    continue

                try:
                    book_output = BookOutput(book_path)
                except Exception as error:
                    logging.error("Failed [%s]: %s", Path(book_path).name, error)
                    fail_count += 1
                    progress.update(1)
                    continue
                book_outputs[book_path] = book_output
                try:
                    for batch_start in range(0, len(image_paths), args.batch_size):
                        while (
                            queued_batch_count(pending, book_outputs)
                            >= max_pending_batches
                        ):
                            succeeded, failed = drain_postprocess_tasks(
                                pending,
                                book_outputs,
                                progress,
                                block=True,
                            )
                            success_count += succeeded
                            fail_count += failed

                        batch_paths = image_paths[
                            batch_start : batch_start + args.batch_size
                        ]
                        raw_results = list(
                            detector.predict_raw_iter(
                                batch_paths,
                                batch_size=len(batch_paths),
                            )
                        )
                        batch_index = book_output.submitted_batches
                        future = postprocess_executor.submit(
                            postprocess_batch_task,
                            (
                                book_path,
                                batch_index,
                                raw_results,
                                args.threshold,
                            ),
                        )
                        registered_batch = book_output.register_batch()
                        if registered_batch != batch_index:
                            raise RuntimeError(
                                "Book batch registration is out of order"
                            )
                        pending[future] = (book_path, batch_index)

                        succeeded, failed = drain_postprocess_tasks(
                            pending,
                            book_outputs,
                            progress,
                            block=False,
                        )
                        success_count += succeeded
                        fail_count += failed
                except Exception as error:
                    logging.exception("Prediction failed [%s]", Path(book_path).name)
                    book_output.seal(error)
                else:
                    book_output.seal()
                succeeded, failed = drain_postprocess_tasks(
                    pending,
                    book_outputs,
                    progress,
                    block=False,
                )
                success_count += succeeded
                fail_count += failed

            # Prediction is complete. Release GPU memory while CPU workers drain.
            detector.close()
            detector = None

            while pending:
                succeeded, failed = drain_postprocess_tasks(
                    pending,
                    book_outputs,
                    progress,
                    block=True,
                )
                success_count += succeeded
                fail_count += failed

    except BaseException:
        if detector is not None:
            detector.close()
        shutdown_executor(postprocess_executor, pending, terminate=True)
        executor_shutdown = True
        raise
    else:
        shutdown_executor(postprocess_executor, pending)
        executor_shutdown = True
    finally:
        if detector is not None:
            detector.close()
        for book_output in book_outputs.values():
            book_output.close()
        if not executor_shutdown:
            shutdown_executor(postprocess_executor, pending, terminate=True)

    print("\n--- Detection Pipeline Complete ---")
    print(f"Successfully annotated: {success_count} books")
    print(f"Failed: {fail_count} books")
    print("Check detection_pipeline.log for detailed error traces.")


def handle_sigterm(signum, frame):
    """Convert termination signals into the pipeline's cleanup path."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Predictor and workers have been released.")
        sys.exit(130)
