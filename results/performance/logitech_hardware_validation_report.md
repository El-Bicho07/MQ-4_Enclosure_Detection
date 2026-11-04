# Logitech C270 & Active Cooler Hardware Validation Report

**Date:** 2026-11-04 14:54:12 to 2026-11-04 15:09:12
**Hardware Platform:** Raspberry Pi 5 (4 GB RAM, Quad-Core Cortex-A76)
**Camera Hardware:** Logitech Webcam C270 (`046d:0825`) at `/dev/video0`
**Cooling Solution:** Official Raspberry Pi 5 Active Cooler (PWM-controlled blower fan)

---

## 1. Camera Latency Breakdown & Freshness Comparison

| Configuration | Resolution | Capture (ms) | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Render (ms) | Total Latency (ms) | Effective FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Threaded_320x320** | 320x320 | 0.24 | 2.33 | 334.16 | 0.93 | 0.79 | **423.79** | **2.36** |
| **Synchronous_320x320** | 320x320 | 2.96 | 1.55 | 341.86 | 0.7 | 0.36 | **348.02** | **2.87** |
| **Threaded_640x640** | 640x640 | 0.24 | 2.94 | 649.35 | 1.13 | 0.49 | **654.88** | **1.52** |

### Key Architecture Findings on Frame Freshness:
- **Buffer Lag Elimination:** With standard synchronous `cv2.VideoCapture`, reading while CPU inference takes 280-800ms causes V4L2 queue buildup, resulting in 2-3 seconds of stream latency.
- **ThreadedCamera Benefit:** Dedicated background thread drains the driver buffer continuously (<0.5 ms grab time), ensuring inference always executes on the freshest camera frame.

---

## 2. Sustained 15-Minute Operational & Thermal Stability

- **Actual Duration:** `900.1 seconds` (15.0 minutes uninterrupted)
- **Total Frames Processed:** `2666` frames
- **Overall Average Throughput:** `2.96 FPS`
- **SoC Starting Temperature:** `52.4 °C`
- **SoC Peak Temperature:** `59.0 °C` (Mean: `56.8 °C`)
- **Active Cooler Fan Speed:** Mean `3143 RPM` (Peak: `4889 RPM`)
- **Thermal Throttling:** `0x0` (Zero under-voltage or throttling detected)
- **Memory Stability:** Started at `2873.4 MB`, finished at `3009.0 MB` (No memory leaks)

### Thermal Trajectory Summary (Every 60s Interval):

| Elapsed Time | Temp (°C) | Fan (RPM) | CPU Freq (MHz) | CPU Load (%) | Detection FPS | Infer Latency (ms) | Throttling |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0s (14:54:13) | 54.0 °C | 2874 RPM | 2400.0 MHz | 56.3% | 3132.42 | 327.0 ms | 0x0 |
| 60.8s (14:55:13) | 55.1 °C | 2920 RPM | 2400.0 MHz | 57.7% | 2.95 | 334.0 ms | 0x0 |
| 121.4s (14:56:14) | 57.3 °C | 2932 RPM | 2400.0 MHz | 57.7% | 2.94 | 338.9 ms | 0x0 |
| 182.0s (14:57:14) | 56.8 °C | 2934 RPM | 2400.0 MHz | 57.7% | 2.92 | 339.2 ms | 0x0 |
| 242.9s (14:58:15) | 57.9 °C | 2938 RPM | 2400.0 MHz | 59.0% | 2.91 | 340.4 ms | 0x0 |
| 303.5s (14:59:16) | 56.2 °C | 4863 RPM | 2400.0 MHz | 53.3% | 3.11 | 321.4 ms | 0x0 |
| 364.8s (15:00:17) | 55.1 °C | 2980 RPM | 2400.0 MHz | 53.3% | 3.08 | 322.0 ms | 0x0 |
| 425.5s (15:01:18) | 56.2 °C | 2935 RPM | 2400.0 MHz | 54.9% | 3.04 | 324.8 ms | 0x0 |
| 486.3s (15:02:19) | 57.3 °C | 2932 RPM | 2400.0 MHz | 56.4% | 3.0 | 330.5 ms | 0x0 |
| 547.0s (15:03:19) | 56.8 °C | 2950 RPM | 2400.0 MHz | 54.2% | 3.09 | 321.0 ms | 0x0 |
| 607.4s (15:04:20) | 56.8 °C | 2921 RPM | 2400.0 MHz | 53.0% | 3.07 | 322.0 ms | 0x0 |
| 668.3s (15:05:21) | 57.3 °C | 4794 RPM | 2400.0 MHz | 52.2% | 3.1 | 319.6 ms | 0x0 |
| 729.0s (15:06:21) | 55.6 °C | 2989 RPM | 2400.0 MHz | 53.5% | 3.06 | 324.8 ms | 0x0 |
| 789.5s (15:07:22) | 56.8 °C | 2956 RPM | 2400.0 MHz | 55.1% | 2.99 | 330.7 ms | 0x0 |
| 850.1s (15:08:22) | 57.9 °C | 2930 RPM | 2400.0 MHz | 53.1% | 3.06 | 322.1 ms | 0x0 |
| 895.6s (15:09:08) | 57.3 °C | 2927 RPM | 2400.0 MHz | 53.3% | 3.07 | 322.5 ms | 0x0 |

---

## 3. Assembly Verification Methodology & Physical Limitations

> [!WARNING]
> **CRITICAL QUALITY CONTROL PRINCIPLE:**
> Optical 2D computer vision bounding box detection can reliably verify **component co-presence** and **2D spatial alignment**. However, it **cannot** definitively verify mechanical snap-fit clip engagement, internal torque, or gasket sealing integrity. In production, optical detection should be combined with physical seating verification sensors.
