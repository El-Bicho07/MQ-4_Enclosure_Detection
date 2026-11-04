# MQ-4 Sensor Enclosure Dataset Specification

## 1. Dataset Overview

- **Dataset Identifier:** `MQ-4_Dataset_v1_Baseline`
- **Annotation Format:** YOLOv8 (normalized `class_id center_x center_y width height`)
- **Total Labeled Images:** 712
- **Color Space:** RGB
- **Native Resolution:** Variable (standardized to 640 × 640 for training and validation)

## 2. Dataset Split Distribution

The dataset was partitioned using stratified sampling to maintain class balance across splits:

| Partition | Image Count | Percentage | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 498 | 70% | Model parameter optimization via SGD/AdamW |
| **Validation** | 142 | 20% | Hyperparameter tuning and checkpoint selection (`best.pt`) |
| **Test** | 72 | 10% | Held-out unbiased model evaluation (Colab test metrics) |
| **Total** | **712** | **100%** | Complete dataset baseline |

## 3. Target Classes & Annotation Guidelines

| Class ID | Class Label | Description | Visual Characteristics |
| :---: | :--- | :--- | :--- |
| `0` | `Enclosure` | Assembled / Full Housing | Complete MQ-4 enclosure assembly containing sensor body and casing. |
| `1` | `Enclosure_Bottom` | Mounting Base | Lower housing bracket with mounting holes, sensor pin cutouts, and base tabs. |
| `2` | `Enclosure_Top` | Protective Hood / Cap | Upper housing component with cylindrical gas sensor mesh opening and ventilation slits. |

## 4. Dataset Directory Layout (Recommended)

When integrating the raw training or evaluation dataset into this repository, organize files according to standard Ultralytics YOLO layout:

```text
dataset/
├── README.md                  # This specification document
├── data.yaml                  # YOLOv8 dataset configuration
├── images/
│   ├── train/                 # 498 training images
│   ├── val/                   # 142 validation images
│   └── test/                  # 72 held-out test images
└── labels/
    ├── train/                 # 498 YOLO format annotation files
    ├── val/                   # 142 YOLO format annotation files
    └── test/                  # 72 YOLO format annotation files
```

## 5. Sample `data.yaml` Configuration

```yaml
path: /home/team5/MQ-4_Enclosure_Detection/dataset
train: images/train
val: images/val
test: images/test

names:
  0: Enclosure
  1: Enclosure_Bottom
  2: Enclosure_Top
```

## 6. Environmental Variations Captured

The baseline dataset captures typical industrial assembly workbench conditions:
- **Lighting:** Controlled workbench fluorescent lighting, direct halogen, and ambient daylight variations.
- **Angles:** Top-down orthogonal (90°), oblique (45°), and low-angle perspective shots.
- **Occlusions:** Minor hand placement, assembly tool proximity (screwdrivers, calipers), and partial assembly overlap.
- **Backgrounds:** ESD mats, metal workbench surfaces, 3D printer beds, and breadboards.
