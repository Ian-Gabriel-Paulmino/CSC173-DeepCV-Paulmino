# CSC173 Deep Computer Vision Project Progress Report
**Student:** [Ian Gabriel Paulmino], [2022-1729]  
**Date:** December 16, 2025  
**Repository:** https://github.com/yourusername/CSC173-DeepCV-Paulmino



## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation |  Completed | 3,700 bird + 362 cloud/negative samples |
| Negative Sample Integration | Completed | Split negatives by dataset ratio, added empty labels |
| Model Training | Completed | 74 epochs (stopped at epoch 54 best) |
| Test Set Evaluation | Completed | Final metrics on unseen data |




## 1. Dataset

### Sources
- **Bird Images:** [Roboflow Birds-detect v16](https://universe.roboflow.com/detectiondanimaux/birds-detect/dataset/16) (3,700 total images)
  - OBB format (Oriented Bounding Boxes)
  - Original split: 86% train, 6% valid, 8% test
  
- **Negative Samples:** [Roboflow Clouds Dataset v2](https://universe.roboflow.com/clouds-dxrpg/my-first-project-makba/dataset/2) (362 images)
  - Used 60 cloud/sky images as negative samples
  - Original split: 90% train, 10% valid

### Final Dataset Composition
- **Total images:** 3,700 bird + 362 cloud
- **Negative sample strategy:** 
  - Split negatives using dataset ratio 
  - Created empty `.txt` labels for YOLO compatibility
  - Integrated directly into existing folders



## 2. Model Training

### Configuration
- **Model:** YOLOv11n-OBB (Nano, 2.66M parameters)
- **Optimizer:** AdamW (lr=0.002)
- **Learning Rate Decay:** Cosine annealing (lr0=0.001, lrf=0.01)
- **Batch Size:** 16
- **Epochs:** 150 → Stopped at 74 (early stopping at epoch 54)
- **Augmentations:** Mosaic, mixup, ±180° rotation, HSV jittering

### 3. Initial Results

#### Best Model (Epoch 54)
| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.893 |
| **mAP@0.5:0.95** | 0.523 |
| **Precision** | 0.877 |
| **Recall** | 0.825 |

#### Test Set Evaluation (Unseen Data)
| Metric | Value | Generalization |
|--------|-------|-----------------|
| **mAP@0.5** | 0.881 | -1.2%  |
| **mAP@0.5:0.95** | 0.523 | 0.0%  |
| **Precision** | 0.822 | -5.5% |
| **Recall** | 0.850 | +2.5% |
| **Inference Speed** | 3.6 ms/image | ~270 FPS |



##  Next Steps

- [ ] Create inference demo script
- [ ] Generate test set predictions visualization
- [ ] Write comprehensive README.md
- [ ] Record 5-minute demo video
- [ ] Push code to GitHub repository


## Dataset Links

- **Bird Images:** https://universe.roboflow.com/detectiondanimaux/birds-detect/dataset/16
- **Negative Samples (Clouds):** https://universe.roboflow.com/clouds-dxrpg/my-first-project-makba/dataset/2
