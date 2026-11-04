"""Image inference module for MQ-4 Enclosure Detection.

Performs YOLOv8n object detection on static images, logs detected components,
and saves annotated visualizations under results/predictions/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from src.config import DetectorConfig, get_project_root
from src.detector import DetectionResult, MQ4Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ImageInference")


def run_image_inference(
    image_path: Path | str,
    detector: Optional[MQ4Detector] = None,
    config: Optional[DetectorConfig] = None,
    conf_thresh: Optional[float] = None,
    iou_thresh: Optional[float] = None,
    imgsz: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
    save_annotated: bool = True,
    show_window: bool = False,
) -> Tuple[DetectionResult, Optional[Path], np.ndarray]:
    """Execute object detection on a single image file.

    Args:
        image_path: Path to the input image file.
        detector: Optional pre-loaded MQ4Detector instance.
        config: Optional DetectorConfig instance.
        conf_thresh: Confidence threshold override.
        iou_thresh: IoU threshold override.
        imgsz: Inference image size override (e.g. 640, 480, 320).
        output_dir: Custom output directory for annotated images.
        save_annotated: Whether to save annotated image to disk.
        show_window: Whether to display OpenCV GUI window (requires display).

    Returns:
        Tuple of (DetectionResult, Optional[saved_image_path], annotated_image_np).

    Raises:
        FileNotFoundError: If the input image is not found.
        ValueError: If the image cannot be read by OpenCV.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    # Load detector if not provided
    cfg = config or DetectorConfig()
    active_detector = detector or MQ4Detector(config=cfg)

    # Read image with OpenCV
    raw_image = cv2.imread(str(path))
    if raw_image is None:
        raise ValueError(f"OpenCV failed to decode image at '{path}'. Verify image format.")

    orig_h, orig_w = raw_image.shape[:2]
    logger.info("Processing image: %s (Resolution: %dx%d)", path.name, orig_w, orig_h)

    # Run inference
    result = active_detector.predict(
        source=raw_image,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
    )

    # Annotate image
    annotated = active_detector.draw_annotations(raw_image, result)

    # Log summary
    print("\n" + "=" * 60)
    print(f"MQ-4 ENCLOSURE DETECTION RESULTS: {path.name}")
    print("=" * 60)
    print(f"Resolution:     {orig_w} x {orig_h}")
    print(f"Components:     {len(result)} detected")
    print(
        f"Speed:          Preprocess {result.speed.get('preprocess', 0):.1f}ms | "
        f"Inference {result.speed.get('inference', 0):.1f}ms | "
        f"Postprocess {result.speed.get('postprocess', 0):.1f}ms"
    )
    print(f"Total Latency:  {result.total_latency_ms:.1f}ms (~{result.fps:.1f} FPS)")
    print("-" * 60)

    if result.is_empty:
        print("  [No enclosure components detected above threshold]")
    else:
        for idx, (cls_name, score, box) in enumerate(
            zip(result.class_names, result.scores, result.boxes), start=1
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            print(
                f"  [{idx}] Class: {cls_name:<18} | Conf: {score * 100:5.1f}% | Box: [{x1}, {y1}, {x2}, {y2}]"
            )
    print("=" * 60 + "\n")

    saved_path: Optional[Path] = None
    if save_annotated:
        out_directory = (
            Path(output_dir).resolve() if output_dir else cfg.predictions_dir
        )
        out_directory.mkdir(parents=True, exist_ok=True)
        saved_filename = f"{path.stem}_pred{path.suffix if path.suffix else '.jpg'}"
        saved_path = out_directory / saved_filename
        cv2.imwrite(str(saved_path), annotated)
        logger.info("Saved annotated prediction to: %s", saved_path)

    if show_window:
        try:
            cv2.imshow("MQ-4 Enclosure Detection", annotated)
            logger.info("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as cv_err:
            logger.warning(
                "Could not open GUI window (headless/no DISPLAY?): %s", cv_err
            )

    return result, saved_path, annotated


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for image inference."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n inference on an image for MQ-4 enclosure detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to input image file (JPEG, PNG, etc.).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional custom path to model weights (.pt).",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=None,
        help="Confidence threshold (0.0 to 1.0). Default is config value (0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold (0.0 to 1.0). Default is 0.45.",
    )
    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=None,
        help="Inference image resolution (e.g. 640, 480, 320).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save annotated prediction images.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display image in OpenCV GUI window (requires graphical display).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    config = DetectorConfig()
    if args.model:
        config.model_path = Path(args.model)

    try:
        run_image_inference(
            image_path=args.image,
            config=config,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            imgsz=args.imgsz,
            output_dir=args.output_dir,
            save_annotated=not args.no_save,
            show_window=args.show,
        )
    except Exception as exc:
        logger.error("Execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
