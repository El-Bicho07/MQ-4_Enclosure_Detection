"""Live camera stream inference and diagnostic module for MQ-4 Enclosure Detection.

Engineered for Logitech USB webcams (e.g. Logitech C270) and Raspberry Pi 5.
Supports V4L2 device auto-discovery via POSIX ioctl, threaded zero-lag capture,
fine-grained latency profiling, spatial assembly validation, and headless operation.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import logging
import os
import signal
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from src.config import DetectorConfig
from src.detector import DetectionResult, MQ4Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CameraInference")

# V4L2 ioctl definition: _IOR('V', 0, struct v4l2_capability)
VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_DEVICE_CAPS = 0x80000000


def is_gui_available() -> bool:
    """Check whether a graphical display server (X11 or Wayland) is reachable."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_v4l2_device_info(device_path: str) -> Optional[Dict[str, Any]]:
    """Query low-level V4L2 capabilities using POSIX ioctl to identify true video capture nodes.

    Filters out metadata nodes (e.g. /dev/video1 on Logitech C270) and internal
    Raspberry Pi ISP/codec nodes (video19-video35) without blocking or timing out.

    Args:
        device_path: Path to character device node, e.g. '/dev/video0'.

    Returns:
        Dictionary of device properties if valid video capture node, else None.
    """
    if not os.path.exists(device_path):
        return None

    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except (OSError, PermissionError):
        return None

    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        driver, card, bus, ver, caps, dev_caps = struct.unpack("16s32s32sIII12x", buf[:104])

        effective_caps = dev_caps if (caps & V4L2_CAP_DEVICE_CAPS) else caps
        is_capture = bool(effective_caps & V4L2_CAP_VIDEO_CAPTURE)

        card_str = card.decode("latin1", errors="replace").strip("\x00")
        driver_str = driver.decode("latin1", errors="replace").strip("\x00")

        # Exclude internal Raspberry Pi 5 ISP backend and HEVC decoder nodes
        lower_card = card_str.lower()
        if any(ign in lower_card for ign in ("pispbe", "pisp_be", "codec", "hevc", "rpivid")):
            return None

        if not is_capture:
            return None

        return {
            "device": device_path,
            "card": card_str,
            "driver": driver_str,
            "caps": hex(effective_caps),
            "is_capture": True,
        }
    except Exception:
        return None
    finally:
        os.close(fd)


def list_available_cameras() -> List[Dict[str, Any]]:
    """Scan and probe system video devices to identify working capture cameras.

    Uses POSIX V4L2 querycap to safely skip non-capture metadata and codec nodes.

    Returns:
        List of dictionaries with device index/path, card name, resolution, and FPS.
    """
    found_devices: List[Dict[str, Any]] = []
    video_nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(p.replace("/dev/video", "")) if p.replace("/dev/video", "").isdigit() else 999,
    )

    for node in video_nodes:
        info = get_v4l2_device_info(node)
        if info is None:
            continue

        # Test opening with OpenCV using V4L2 backend
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            cap.release()

            dev_idx = int(node.replace("/dev/video", ""))
            found_devices.append({
                "device_index": dev_idx,
                "device_path": node,
                "card_name": info["card"],
                "driver": info["driver"],
                "width": w,
                "height": h,
                "fps": fps if fps > 0 else 30.0,
                "can_read_frame": ret,
            })

    return found_devices


def test_camera_device(
    device: Union[int, str] = 0,
    save_test_image: bool = True,
    output_path: Optional[Path | str] = None,
    warmup_frames: int = 5,
) -> Tuple[bool, str, Optional[Path]]:
    """Quick hardware diagnostic utility to verify camera capture and save a test frame.

    Operates without loading the YOLO model for instantaneous hardware verification.

    Args:
        device: Camera index (int) or device path (str).
        save_test_image: Whether to write a sample captured frame to disk.
        output_path: Custom destination for test image.
        warmup_frames: Number of initial frames to discard for auto-exposure stabilization.

    Returns:
        Tuple of (success_bool, diagnostic_message, Optional[saved_image_path]).
    """
    dev_target: Union[int, str] = int(device) if isinstance(device, str) and device.isdigit() else device

    logger.info("Running hardware diagnostic on camera: %s", dev_target)
    cap = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)

    if not cap.isOpened():
        # Fallback to default backend if CAP_V4L2 failed
        cap = cv2.VideoCapture(dev_target)

    if not cap.isOpened():
        msg = f"Camera device '{dev_target}' could not be opened. Check hardware connection."
        logger.error(msg)
        return False, msg, None

    try:
        # Request MJPG for Logitech webcams if supported
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Discard warmup frames so sensor auto-exposure and white-balance settle
        frame = None
        for _ in range(max(1, warmup_frames)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        if frame is None:
            msg = f"Camera '{dev_target}' opened but failed to deliver a valid frame."
            logger.error(msg)
            return False, msg, None

        saved_path: Optional[Path] = None
        if save_test_image:
            out_file = (
                Path(output_path).resolve()
                if output_path
                else DetectorConfig().predictions_dir / "camera_test_frame.jpg"
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_file), frame)
            saved_path = out_file
            logger.info("Diagnostic frame successfully saved to: %s", out_file)

        msg = (
            f"Camera '{dev_target}' is operational: Native {w}x{h} @ ~{fps:.1f} FPS. "
            f"Captured frame shape: {frame.shape}."
        )
        logger.info(msg)
        return True, msg, saved_path

    finally:
        cap.release()


class ThreadedCamera:
    """Dedicated background worker thread for non-blocking camera frame acquisition.

    Continuously drains the V4L2 kernel driver queue so the application
    always processes the freshest possible frame without multi-second buffer lag.
    """

    def __init__(
        self,
        src: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        fourcc: str = "MJPG",
    ) -> None:
        """Initialize and start background capture thread."""
        dev_target: Union[int, str] = int(src) if isinstance(src, str) and src.isdigit() else src

        # Prefer V4L2 backend on Linux
        self.cap = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev_target)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source '{src}'.")

        # Set hardware properties
        if fourcc and len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.stopped = False
        self.lock = threading.Lock()
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        self.grab_count = 0
        self.latest_grab_timestamp = 0.0

        # Prime initial frame
        self.ret, self.frame = self.cap.read()
        if self.ret and self.frame is not None:
            self.grab_count = 1
            self.latest_grab_timestamp = time.perf_counter()

        # Start worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        """Background loop continuously polling newest camera frame."""
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret and frame is not None:
                    self.frame = frame
                    self.grab_count += 1
                    self.latest_grab_timestamp = time.perf_counter()
                else:
                    self.stopped = True
                    break

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Return the most recently acquired frame and its acquisition latency.

        Returns:
            Tuple of (ret_bool, frame_copy_or_None, grab_latency_ms).
        """
        t0 = time.perf_counter()
        with self.lock:
            if self.frame is not None:
                frame_out = self.frame.copy()
                grab_latency_ms = (time.perf_counter() - t0) * 1000.0
                return self.ret, frame_out, grab_latency_ms
            return self.ret, None, 0.0

    def release(self) -> None:
        """Stop background reader and release video capture resources."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    return (inter_area / union_area) if union_area > 0.0 else 0.0


def compute_assembly_status(result: DetectionResult) -> Tuple[str, Tuple[int, int, int], str]:
    """Evaluate enclosure assembly status considering class identities and spatial relationships.

    CRITICAL QUALITY CONTROL NOTE:
    Optical 2D object detection cannot guarantee internal mechanical clip latching
    or seal integrity. It evaluates visual alignment and component co-presence only.

    Args:
        result: DetectionResult from YOLO inference.

    Returns:
        Tuple of (status_title, BGR_badge_color, detailed_diagnostic_string).
    """
    has_full = any(cls_id == 0 for cls_id in result.class_ids)
    bottom_indices = [i for i, cid in enumerate(result.class_ids) if cid == 1]
    top_indices = [i for i, cid in enumerate(result.class_ids) if cid == 2]

    # Primary case: Assembled unit detected as single monolithic assembly
    if has_full:
        full_idx = result.class_ids.index(0)
        conf = result.scores[full_idx] * 100.0
        return (
            "PASS: ASSEMBLED ENCLOSURE (OPTICAL)",
            (50, 205, 50),  # Green
            f"Class 0 detected ({conf:.1f}% conf). Visual assembly verified.",
        )

    # Secondary case: Both base and top sub-components detected in frame
    if bottom_indices and top_indices:
        # Check spatial proximity between best bottom and top
        best_b_idx = max(bottom_indices, key=lambda i: result.scores[i])
        best_t_idx = max(top_indices, key=lambda i: result.scores[i])

        box_b = result.boxes[best_b_idx]
        box_t = result.boxes[best_t_idx]
        iou = compute_box_iou(box_b, box_t)

        # Check vertical arrangement: Top cap should sit above or overlap with base plate
        b_center_y = (box_b[1] + box_b[3]) / 2.0
        t_center_y = (box_t[1] + box_t[3]) / 2.0

        if iou > 0.15 or (t_center_y < b_center_y and iou > 0.02):
            return (
                "NOTICE: CANDIDATE ASSEMBLY (VERIFY LATCH)",
                (0, 215, 255),  # Gold
                f"Base and Cap spatially aligned (IoU: {iou:.2f}). Check physical snap-fit.",
            )
        else:
            return (
                "WARN: DETACHED COMPONENTS IN FRAME",
                (0, 165, 255),  # Orange
                f"Base & Cap visible but unaligned (IoU: {iou:.2f}). Assembly incomplete.",
            )

    # Sub-component cases
    if bottom_indices:
        return (
            "SUB-ASSEMBLY: ENCLOSURE_BOTTOM ONLY",
            (255, 180, 0),  # Cyan
            "Lower base plate present; top protective cap is missing.",
        )

    if top_indices:
        return (
            "SUB-ASSEMBLY: ENCLOSURE_TOP ONLY",
            (0, 140, 255),  # Amber
            "Top cap present; lower mounting base is missing.",
        )

    return (
        "READY: NO ENCLOSURE DETECTED",
        (160, 160, 160),  # Neutral Grey
        "Inspection workstation clear. Place unit in camera field of view.",
    )


def run_camera_stream(
    device: Union[int, str] = 0,
    detector: Optional[MQ4Detector] = None,
    config: Optional[DetectorConfig] = None,
    conf_thresh: Optional[float] = None,
    iou_thresh: Optional[float] = None,
    imgsz: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    headless: bool = False,
    threaded: bool = True,
    grayscale: bool = False,
    record_path: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
) -> int:
    """Run real-time object detection stream from a camera device.

    Args:
        device: Camera index (int, e.g. 0) or device node path (str, e.g. '/dev/video0').
        detector: Optional pre-loaded MQ4Detector instance.
        config: Optional DetectorConfig instance.
        conf_thresh: Confidence threshold override.
        iou_thresh: IoU threshold override.
        imgsz: Inference image size override (e.g. 640, 480, 320).
        width: Desired camera capture frame width.
        height: Desired camera capture frame height.
        headless: If True, suppress OpenCV GUI window and log to console.
        threaded: If True, use background thread for lag-free frame acquisition.
        grayscale: If True, process frame in grayscale.
        record_path: Optional output video file path to record stream.
        max_frames: Optional frame limit for testing or batch acquisition.

    Returns:
        Total number of processed frames.

    Raises:
        RuntimeError: If camera cannot be initialized.
    """
    cfg = config or DetectorConfig()
    active_detector = detector or MQ4Detector(config=cfg)

    # Determine if GUI is viable
    gui_mode = (not headless) and is_gui_available()
    if not gui_mode and not headless:
        logger.warning(
            "No graphical display detected ($DISPLAY / $WAYLAND_DISPLAY unset). "
            "Running camera stream in HEADLESS mode (console output only)."
        )

    target_w = width or cfg.camera_width
    target_h = height or cfg.camera_height

    cam: Union[ThreadedCamera, cv2.VideoCapture]
    actual_w: int
    actual_h: int
    actual_fps: float

    try:
        if threaded:
            logger.info("Initializing ThreadedCamera on device: %s (Format: MJPG)", device)
            cam = ThreadedCamera(src=device, width=target_w, height=target_h, fps=cfg.camera_fps)
            actual_w = cam.actual_width
            actual_h = cam.actual_height
            actual_fps = cam.actual_fps
        else:
            dev_target = int(device) if isinstance(device, str) and device.isdigit() else device
            logger.info("Initializing standard VideoCapture on device: %s", dev_target)
            cam = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
            if not cam.isOpened():
                cam = cv2.VideoCapture(dev_target)
            if not cam.isOpened():
                raise RuntimeError(f"Could not open camera source '{device}'.")

            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cam.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
            actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cam.get(cv2.CAP_PROP_FPS)

    except Exception as exc:
        error_msg = (
            f"Failed to initialize camera source '{device}': {exc}\n"
            "Troubleshooting Steps for Raspberry Pi:\n"
            "  1. Run 'python3 src/camera.py --list-devices' to scan active video nodes.\n"
            "  2. Run 'python3 src/camera.py --test-camera --device <index>' to test without YOLO.\n"
            "  3. Check USB permissions: ensure current user is in 'video' group ('groups').\n"
            "  4. Check hardware status: 'lsusb' and 'v4l2-ctl --list-devices'."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc

    logger.info(
        "Camera stream ready: %dx%d @ ~%.1f FPS (Requested: %dx%d)",
        actual_w,
        actual_h,
        actual_fps,
        target_w,
        target_h,
    )

    # Initialize video writer if recording requested
    video_writer: Optional[cv2.VideoWriter] = None
    if record_path:
        rec_path = Path(record_path).resolve()
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_to_use = actual_fps if actual_fps > 0 else 30.0
        video_writer = cv2.VideoWriter(
            str(rec_path), fourcc, fps_to_use, (actual_w, actual_h)
        )
        logger.info("Recording stream to: %s", rec_path)

    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        logger.info("Termination signal received. Stopping camera stream...")
        interrupted = True

    prev_sigint = signal.signal(signal.SIGINT, handle_sigint)

    fps_buffer = deque(maxlen=30)
    frame_count = 0
    consecutive_failures = 0
    start_time = time.time()
    paused = False
    show_hud = True
    last_annotated: Optional[np.ndarray] = None

    logger.info("=" * 60)
    logger.info("MQ-4 CAMERA DETECTION ACTIVE")
    if gui_mode:
        logger.info("  [q] / [ESC]: Quit Stream")
        logger.info("  [s]:         Save Evidence Snapshot to results/predictions/")
        logger.info("  [p]:         Pause / Freeze Live Stream")
        logger.info("  [h]:         Toggle On-Screen HUD Overlay")
    else:
        logger.info("  Running in HEADLESS mode. Press Ctrl+C to terminate.")
    logger.info("=" * 60)

    try:
        while not interrupted:
            t_loop_start = time.perf_counter()

            if not paused:
                # Capture frame
                t_grab_start = time.perf_counter()
                if threaded:
                    ret, frame, _ = cam.read()
                else:
                    ret, frame = cam.read()
                t_grab_ms = (time.perf_counter() - t_grab_start) * 1000.0

                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(
                        "Frame acquisition failure (%d/5). Camera disconnected?",
                        consecutive_failures,
                    )
                    if consecutive_failures >= 5:
                        logger.error("Camera disconnected or unresponsive. Terminating stream.")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                frame_count += 1

                if grayscale:
                    gray_single = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray_single, cv2.COLOR_GRAY2BGR)

                # Execute YOLO inference
                result = active_detector.predict(
                    source=frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=imgsz,
                )

                # Render detection boxes
                t_render_start = time.perf_counter()
                annotated = active_detector.draw_annotations(
                    frame, result, show_fps=False
                )

                status_text, status_color, status_detail = compute_assembly_status(result)

                if show_hud:
                    banner_h = 32
                    cv2.rectangle(annotated, (0, 0), (actual_w, banner_h), (25, 25, 25), -1)
                    cv2.putText(
                        annotated,
                        status_text,
                        (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        status_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )

                    t_loop_elapsed = time.perf_counter() - t_loop_start
                    fps_buffer.append(t_loop_elapsed)
                    effective_fps = len(fps_buffer) / sum(fps_buffer) if fps_buffer else 0.0

                    counts_summary = (
                        f"Enclosure: {result.class_ids.count(0)} | "
                        f"Base: {result.class_ids.count(1)} | "
                        f"Cap: {result.class_ids.count(2)}"
                    )
                    hud_text = (
                        f"Detection FPS: {effective_fps:4.1f} | "
                        f"Infer: {result.speed.get('inference', 0.0):5.1f}ms | "
                        f"Grab: {t_grab_ms:4.1f}ms | {counts_summary}"
                    )
                    cv2.rectangle(
                        annotated,
                        (0, actual_h - 26),
                        (actual_w, actual_h),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        hud_text,
                        (10, actual_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (0, 255, 200),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                last_annotated = annotated
            else:
                if last_annotated is not None:
                    annotated = last_annotated.copy()
                    cv2.putText(
                        annotated,
                        "[STREAM PAUSED - PRESS 'p' TO RESUME]",
                        (actual_w // 2 - 200, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                time.sleep(0.03)

            # Record frame if active
            if video_writer is not None and last_annotated is not None:
                video_writer.write(last_annotated)

            # GUI display or headless logging
            if gui_mode and last_annotated is not None:
                cv2.imshow("MQ-4 Enclosure Real-Time Inspection", last_annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Quit key pressed. Stopping stream.")
                    break
                elif key in (ord("p"), ord("P")):
                    paused = not paused
                    logger.info("Stream %s", "PAUSED" if paused else "RESUMED")
                elif key in (ord("h"), ord("H")):
                    show_hud = not show_hud
                    logger.info("HUD overlay %s", "ENABLED" if show_hud else "DISABLED")
                elif key in (ord("s"), ord("S")):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_name = f"snapshot_logitech_{ts}.jpg"
                    snap_path = cfg.predictions_dir / snap_name
                    cv2.imwrite(str(snap_path), last_annotated)
                    logger.info("Saved snapshot evidence to: %s", snap_path)
            else:
                if frame_count % 30 == 0 or (result and not result.is_empty):
                    det_summary = (
                        ", ".join(
                            f"{name} ({score * 100:.1f}%)"
                            for name, score in zip(result.class_names, result.scores)
                        )
                        if result and not result.is_empty
                        else "No objects"
                    )
                    effective_fps = len(fps_buffer) / sum(fps_buffer) if fps_buffer else 0.0
                    logger.info(
                        "[Frame %4d] %s | %s | Eff FPS: %.1f | Infer: %.1fms",
                        frame_count,
                        status_text,
                        det_summary,
                        effective_fps,
                        result.speed.get("inference", 0.0) if result else 0.0,
                    )

            if max_frames and frame_count >= max_frames:
                logger.info("Reached maximum frame limit (%d). Exiting.", max_frames)
                break

    finally:
        signal.signal(signal.SIGINT, prev_sigint)

        logger.info("Cleaning up camera and window resources...")
        cam.release()
        if video_writer is not None:
            video_writer.release()
        if gui_mode:
            cv2.destroyAllWindows()

        total_elapsed = time.time() - start_time
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        logger.info(
            "Stream terminated cleanly. Processed %d frames in %.2f s (Average: %.2f FPS).",
            frame_count,
            total_elapsed,
            avg_fps,
        )

    return frame_count


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for camera stream and diagnostics."""
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8n camera inspection and diagnostics for MQ-4 sensor enclosures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="0",
        help="Camera device index (e.g. 0) or device path (e.g. /dev/video0).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Probe and list all available video camera devices and exit.",
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Run fast hardware diagnostic on target camera (verifies capture and saves test frame).",
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
        help="Confidence threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU NMS threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=None,
        help="Inference image resolution (e.g. 640, 480, 320).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Camera frame capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Camera frame capture height.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening GUI window (recommended for SSH / remote sessions).",
    )
    parser.add_argument(
        "--no-threaded",
        action="store_true",
        help="Disable background threaded frame capture (use standard synchronous reading).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Process camera frames in grayscale mode.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Save annotated video stream to specified file (e.g. results/stream.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process before exiting.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    # Device enumeration utility
    if args.list_devices:
        print("\n" + "=" * 60)
        print("SCANNING FOR CONNECTED CAMERA DEVICES...")
        print("=" * 60)
        cameras = list_available_cameras()
        if not cameras:
            print("  No operational camera capture devices detected.")
            print("  Check physical USB connection, 'lsusb', and user 'video' group.")
        else:
            for cam_info in cameras:
                print(
                    f"  Index: {cam_info['device_index']} | "
                    f"Path: {cam_info['device_path']} | "
                    f"Name: {cam_info['card_name']} | "
                    f"Format: {cam_info['width']}x{cam_info['height']} @ ~{cam_info['fps']:.0f} FPS | "
                    f"Capture OK: {cam_info['can_read_frame']}"
                )
        print("=" * 60 + "\n")
        return

    # Diagnostic mode
    if args.test_camera:
        print("\n" + "=" * 60)
        print(f"TESTING CAMERA DEVICE: {args.device}")
        print("=" * 60)
        ok, msg, snap = test_camera_device(args.device)
        print(f"Status:  {'SUCCESS' if ok else 'FAILED'}")
        print(f"Details: {msg}")
        if snap:
            print(f"Frame:   Saved to {snap}")
        print("=" * 60 + "\n")
        sys.exit(0 if ok else 1)

    # Full Detection Stream
    config = DetectorConfig()
    if args.model:
        config.model_path = Path(args.model)

    try:
        run_camera_stream(
            device=args.device,
            config=config,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            imgsz=args.imgsz,
            width=args.width,
            height=args.height,
            headless=args.headless,
            threaded=not args.no_threaded,
            grayscale=args.grayscale,
            record_path=args.record,
            max_frames=args.max_frames,
        )
    except RuntimeError as r_err:
        logger.error("%s", r_err)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error during camera execution: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
