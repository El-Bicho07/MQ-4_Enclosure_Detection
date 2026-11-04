"""Detection engine for MQ-4 Enclosure Detection using YOLOv8n.

Provides model loading, inference processing, error handling, and visual
annotation for images and live camera streams.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import DEFAULT_CLASS_COLORS, DEFAULT_CLASS_NAMES, DetectorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MQ4Detector")


@dataclass
class DetectionResult:
    """Structured container for object detection outputs."""

    # Bounding boxes in [x1, y1, x2, y2] pixel coordinates
    boxes: List[List[float]] = field(default_factory=list)

    # Confidence scores [0.0 - 1.0]
    scores: List[float] = field(default_factory=list)

    # Class IDs (integers)
    class_ids: List[int] = field(default_factory=list)

    # Class labels (strings)
    class_names: List[str] = field(default_factory=list)

    # Latency breakdown in milliseconds: {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    speed: Dict[str, float] = field(default_factory=dict)

    # Original input image shape (height, width)
    orig_shape: Tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def is_empty(self) -> bool:
        return len(self.boxes) == 0

    @property
    def total_latency_ms(self) -> float:
        """Total processing latency in milliseconds."""
        return sum(self.speed.values()) if self.speed else 0.0

    @property
    def fps(self) -> float:
        """Estimated frames per second based on total latency."""
        total = self.total_latency_ms
        return (1000.0 / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to serializable dictionary."""
        return {
            "count": len(self.boxes),
            "detections": [
                {
                    "box": self.boxes[i],
                    "confidence": round(float(self.scores[i]), 4),
                    "class_id": int(self.class_ids[i]),
                    "class_name": self.class_names[i],
                }
                for i in range(len(self.boxes))
            ],
            "speed_ms": {k: round(v, 2) for k, v in self.speed.items()},
            "total_latency_ms": round(self.total_latency_ms, 2),
            "fps": round(self.fps, 2),
            "orig_shape": self.orig_shape,
        }


class MQ4Detector:
    """YOLOv8n object detector tailored for MQ-4 enclosure components."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """Initialize detector with configuration and load YOLO weights.

        Args:
            config: Optional DetectorConfig instance. If None, default configuration is used.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
            RuntimeError: If model fails to load.
        """
        self.config = config or DetectorConfig()
        self._apply_runtime_optimizations()
        self.model = self._load_model()
        self._validate_model_classes()

    def _apply_runtime_optimizations(self) -> None:
        """Apply CPU and OpenCV runtime performance settings."""
        if self.config.device == "cpu":
            num_threads = max(1, self.config.num_threads)
            torch.set_num_threads(num_threads)
            cv2.setNumThreads(num_threads)
            cv2.setUseOptimized(True)
            logger.info(
                "CPU optimizations active: PyTorch threads=%d, cv2 threads=%d, cv2 optimized=%s",
                num_threads,
                cv2.getNumThreads(),
                cv2.useOptimized(),
            )

    def _load_model(self) -> YOLO:
        """Validate and load the YOLO checkpoint.

        Returns:
            Ultralytics YOLO model instance.
        """
        checkpoint_path = self.config.validate_checkpoint()
        logger.info("Loading YOLOv8n model checkpoint from: %s", checkpoint_path)

        try:
            model = YOLO(str(checkpoint_path))
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOv8 model from '{checkpoint_path}': {exc}"
            ) from exc

    def _validate_model_classes(self) -> None:
        """Verify that loaded model classes match expected classes."""
        model_names = getattr(self.model, "names", {})
        if not model_names:
            logger.warning("Loaded model has no 'names' attribute.")
            return

        logger.info("Model classes identified: %s", model_names)
        expected = self.config.class_names
        for cls_id, expected_name in expected.items():
            loaded_name = model_names.get(cls_id)
            if loaded_name != expected_name:
                logger.warning(
                    "Class ID %d mismatch: expected '%s', got '%s'",
                    cls_id,
                    expected_name,
                    loaded_name,
                )

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> DetectionResult:
        """Run object detection inference on an image or video frame.

        Args:
            source: Image path (str/Path) or numpy array (BGR format).
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.
            imgsz: Optional inference image resolution override.

        Returns:
            DetectionResult container with detections, labels, and latency.

        Raises:
            ValueError: If input source is invalid or unreadable.
        """
        # Validate inputs
        if source is None:
            raise ValueError("Input source cannot be None.")

        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Input image path does not exist: {src_path}")
            if not src_path.is_file():
                raise ValueError(f"Input image path is not a file: {src_path}")

        elif isinstance(source, np.ndarray):
            if source.size == 0 or len(source.shape) not in (2, 3):
                raise ValueError(
                    f"Invalid numpy array input: shape={source.shape}, size={source.size}"
                )

        # Set inference hyperparameters
        conf_thresh = conf if conf is not None else self.config.confidence_threshold
        iou_thresh = iou if iou is not None else self.config.iou_threshold
        img_size = imgsz if imgsz is not None else self.config.image_size

        try:
            results = self.model.predict(
                source=source,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Error during YOLOv8 inference: {exc}") from exc

        if not results:
            return DetectionResult()

        res = results[0]
        orig_shape = res.orig_shape if hasattr(res, "orig_shape") else (0, 0)
        speed = res.speed if hasattr(res, "speed") else {}

        boxes_list: List[List[float]] = []
        scores_list: List[float] = []
        class_ids_list: List[int] = []
        class_names_list: List[str] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_data = res.boxes.xyxy.cpu().numpy()
            confs_data = res.boxes.conf.cpu().numpy()
            clss_data = res.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes_data, confs_data, clss_data):
                cls_id_int = int(cls_id)
                cls_name = self.config.class_names.get(
                    cls_id_int, res.names.get(cls_id_int, f"class_{cls_id_int}")
                )
                boxes_list.append([float(x) for x in box])
                scores_list.append(float(score))
                class_ids_list.append(cls_id_int)
                class_names_list.append(cls_name)

        return DetectionResult(
            boxes=boxes_list,
            scores=scores_list,
            class_ids=class_ids_list,
            class_names=class_names_list,
            speed=speed,
            orig_shape=orig_shape,
        )

    def draw_annotations(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_fps: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Render detection bounding boxes, labels, and stats onto the image.

        Args:
            image: Original BGR numpy image.
            result: DetectionResult from predict().
            show_fps: Whether to draw FPS/latency badge in upper corner.
            show_labels: Whether to draw class names and confidence badges.

        Returns:
            Annotated BGR numpy image copy.
        """
        annotated = image.copy()
        colors = self.config.class_colors

        for box, score, cls_id, cls_name in zip(
            result.boxes, result.scores, result.class_ids, result.class_names
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = colors.get(cls_id, (0, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            if show_labels:
                label_text = f"{cls_name} {score * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )

                # Ensure label badge stays within image boundary
                badge_y1 = max(0, y1 - text_h - 6)
                badge_y2 = y1
                badge_x1 = x1
                badge_x2 = x1 + text_w + 8

                # Draw filled label background
                cv2.rectangle(
                    annotated,
                    (badge_x1, badge_y1),
                    (badge_x2, badge_y2),
                    color,
                    -1,
                )

                # Text color: dark text on light badge, or white on dark
                text_color = (0, 0, 0)
                cv2.putText(
                    annotated,
                    label_text,
                    (badge_x1 + 4, badge_y2 - 4),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        if show_fps and result.speed:
            total_ms = result.total_latency_ms
            fps_val = result.fps
            stats_text = f"FPS: {fps_val:.1f} | Latency: {total_ms:.1f}ms ({result.orig_shape[1]}x{result.orig_shape[0]})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), _ = cv2.getTextSize(stats_text, font, font_scale, thickness)

            # Draw top overlay banner
            cv2.rectangle(annotated, (10, 10), (20 + tw, 20 + th + 6), (20, 20, 20), -1)
            cv2.rectangle(annotated, (10, 10), (20 + tw, 20 + th + 6), (100, 100, 100), 1)
            cv2.putText(
                annotated,
                stats_text,
                (15, 10 + th + 4),
                font,
                font_scale,
                (0, 255, 128),
                1,
                lineType=cv2.LINE_AA,
            )

        return annotated
