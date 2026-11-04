# Assessment Evidence: Model Training

## 1. Training Architecture & Framework

- **Base Architecture:** YOLOv8n (Nano) Object Detector
- **Framework:** Ultralytics YOLO (`v8.0+`)
- **Backbone:** Modified CSPDarknet53 with C2f modules
- **Neck:** PAN-FPN (Path Aggregation Network)
- **Head:** Anchor-free decoupled detection head
- **Pretrained Weights:** `yolov8n.pt` (COCO pre-trained initialization)
- **Parameters:** ~3.01 Million
- **GFLOPs:** ~8.2 GFLOPs @ 640×640

## 2. Training Platform & Environment

- **Platform:** Google Colab Cloud GPU Instance
- **Hardware Accelerator:** NVIDIA Tesla T4 (16 GB GDDR6 VRAM) / A100
- **Host OS:** Ubuntu 22.04 LTS (x86_64)
- **CUDA Version:** 12.x
- **PyTorch Version:** 2.x

## 3. Training Hyperparameters

The model checkpoint stored at `models/best.pt` preserves the following exact training configuration:

```yaml
model: yolov8n.pt
data: /content/datasets/MQ-4_Sensor_Detection-1/data_train.yaml
epochs: 100
batch: 16
imgsz: 640
optimizer: auto (SGD/AdamW)
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
box: 7.5
cls: 0.5
dfl: 1.5
save: true
```

## 4. Checkpoint Verification

Verification on the edge device confirms:
- **File Location:** `models/best.pt`
- **File Size:** ~6.2 MB (6,498,794 bytes)
- **Model Task:** `detect`
- **Output Classes (3):** `{0: 'Enclosure', 1: 'Enclosure_Bottom', 2: 'Enclosure_Top'}`
- **Integrity Status:** Verified loadable by Ultralytics and PyTorch 2.6.
