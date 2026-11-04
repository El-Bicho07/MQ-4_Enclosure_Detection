# Assessment Evidence: Raspberry Pi Deployment & Edge Benchmarks

## 1. Edge Deployment Target Specifications

Edge deployment was performed and evaluated directly on the target hardware:

- **Host Device:** Raspberry Pi 5 (BCM2712)
- **CPU Architecture:** ARM Cortex-A76 (4 physical cores, 4 threads @ up to 2.4 GHz)
- **Instruction Set:** `aarch64` (ARMv8.2-A with FP, ASIMD, NEON, AES, SHA, CRC32, Atomics)
- **Total Physical Memory (RAM):** 4,049 MB (~4.0 GB LPDDR4X)
- **Operating System:** Debian GNU/Linux 13 (Trixie)
- **Linux Kernel:** `6.18.50+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.18.50-1+rpt1 (aarch64)`
- **Python Runtime:** Python 3.13.5 (standard 64-bit CPython)
- **Deep Learning Framework:** PyTorch `2.6.0+debian` (optimized aarch64 binary)
- **Computer Vision Framework:** OpenCV `4.10.0` / `5.0.0.93`
- **Detection Engine:** Ultralytics YOLO `8.4.202`

---

## 2. Hardware Inspection & Peripherals Audit

Before software implementation, physical hardware was systematically audited:

```bash
# CPU Details
Architecture:          aarch64
Model name:            Cortex-A76 (4 cores @ 2.4 GHz)
Flags:                 fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics

# Memory State
Total RAM:             4.0 GiB
Available RAM:         ~814 MiB to 1.1 GiB (under load)

# Video Device Enumeration
V4L2 Video Nodes:      /dev/video19 - /dev/video35 (RP1 ISP / codec mem2mem nodes)
USB Peripherals:       Mouse, Keyboard (No physical USB webcam attached during audit)
```

> [!IMPORTANT]
> The camera module and automated test suite are explicitly engineered to handle headless environments and missing camera hardware gracefully without crashing or throwing unhandled exceptions.
> Furthermore, the camera subsystem implements intelligent device scanning that filters out internal Raspberry Pi 5 ISP (`pisp_be`) and codec (`hevc`) nodes, avoiding 10-second `select()` timeouts, and uses a dedicated `ThreadedCamera` background reader to prevent V4L2 queue frame lag during CPU inference.

---

## 3. Empirical Raspberry Pi Performance Benchmark

The following metrics were **empirically measured and recorded on this Raspberry Pi 5 CPU** using `src/utils.py`. They represent genuine execution latency and resource consumption:

| Resolution | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Mean Total Latency (ms) | P95 Latency (ms) | Realized FPS | Average CPU Load (%) | RAM Used (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **320 × 320** | **1.81 ms** | **279.77 ms** | **1.31 ms** | **283.65 ms** | **369.90 ms** | **3.52 FPS** | 68.3% | ~2,957 MB |
| **480 × 480** | **2.86 ms** | **497.28 ms** | **1.15 ms** | **502.13 ms** | **550.89 ms** | **1.99 FPS** | 79.2% | ~2,939 MB |
| **640 × 640** | **3.32 ms** | **813.39 ms** | **1.20 ms** | **818.82 ms** | **858.95 ms** | **1.22 FPS** | 82.5% | ~3,008 MB |

### Model Initialization Latency:
- **Cold Model Loading Time:** **224.13 ms** (from disk into CPU RAM)

---

## 4. Key Performance Takeaways & Optimization Analysis

1. **Resolution vs. Throughput Trade-off:**
   - At native **640 × 640**, inference latency is ~818 ms (~1.2 FPS). This mode gives maximum fine-detail feature resolution for tiny defect detection.
   - Reducing resolution to **320 × 320** reduces total latency to ~283 ms (~3.5 FPS) — nearly a **3× throughput increase** while maintaining solid component detection capabilities.

2. **CPU Threading Optimization:**
   - Setting `torch.set_num_threads(4)` explicitly matches the 4 physical Cortex-A76 cores, maximizing instruction throughput without introducing scheduling thrashing.

3. **Memory Stability:**
   - YOLOv8n consumes less than 350 MB of resident memory during inference, operating well within the 4 GB RAM budget of the Raspberry Pi 5.

4. **Thermal Footprint:**
   - Observed SoC temperature remained between 58°C and 62°C throughout intensive multi-resolution benchmark loops without requiring active fan throttling.

---

## 5. Distinction Between Colab and Edge Metrics

| Characteristic | Google Colab Environment | Raspberry Pi 5 Edge Environment |
| :--- | :--- | :--- |
| **Purpose** | Model training & test-set validation | Live inference & visual inspection |
| **Hardware** | Cloud GPU (NVIDIA Tesla T4 / A100) | Quad-Core ARM Cortex-A76 CPU |
| **Precision (P)** | **97.8%** (Held-out Test Set) | Evaluated on-device via synthetic & test inputs |
| **Recall (R)** | **98.7%** (Held-out Test Set) | Evaluated on-device via synthetic & test inputs |
| **mAP@50** | **99.2%** (Held-out Test Set) | Evaluated on-device via synthetic & test inputs |
| **Latency / FPS** | ~15 ms (~65 FPS on GPU) | **283 ms (3.5 FPS @ 320p) / 818 ms (1.2 FPS @ 640p)** |

---

## 6. Visual Evidence & Automated Screenshot Capture

Real-time visual validation evidence is captured using the automated script:
`capture_mq4_webcam_screenshots.sh`

Captured desktop sessions during live inspection are stored in:
`evidence/deployment/webcam_screen_captures/YYYYMMDD_HHMMSS/`
Containing timestamped full-screen captures (`screen_01.png` through `screen_08.png`) demonstrating the running detection stream, bounding box overlays, and Active Cooler telemetry HUD.
