# CSC173 Deep Computer Vision Project Proposal
**Student:** Paulmino, Ian Gabriel D. 2022-1729  
**Date:** December 11, 2025

## 1. Project Title
**Precision Bird Detection for Aviation Safety: A Data-Efficient Approach Using YOLO11-OBB and Strategic Negative Sampling**

## 2. Problem Statement
Bird strikes pose a critical safety hazard to aviation, costing the global industry over $900 million annually, with Philippine airports such as NAIA and Laguindingan facing heightened risks due to surrounding wetlands and coastal bird populations. Traditional detection systems like radar are prohibitively expensive for regional airports, while existing computer vision models struggle with false positives and poor localization of birds in arbitrary flight orientations. This project addresses the need for a low-cost yet highly precise bird detection system suitable for resource-constrained airports. By utilizing oriented bounding box (OBB) detection and data-efficient learning strategies, the system aims to reliably detect birds while significantly reducing false alarms from non-target objects.

## 3. Objectives
- Develop a robust bird detection model by fine-tuning **YOLO11-OBB** on a dataset of 467 images (including 15% negative samples) to handle arbitrary flight orientations and minimize false positives.  
- Achieve **>80% mAP@0.5** accuracy despite limited data by applying advanced augmentation strategies such as mosaic, mixup, and 360° rotation, supported by transfer learning.  
- Quantify the impact of negative sampling through an ablation study measuring reductions in false positive rates for confusing scenarios such as empty skies, people, and aircraft.  
- Validate **real-time inference speed (<15ms per frame)** on a standard GPU to propose a proof of concept framework for Philippine regional airports (e.g., Laguindingan, Davao) that utilizes low-cost camera infrastructure and open-source computer vision software to enhance aviation safety..  
- Implement an end-to-end pipeline including dataset preparation (with negative samples), training, validation, benchmark comparison, and video inference demonstration.

## 4. Dataset Plan
**Source:**  
Bird Detection with Oriented Bounding Boxes (Roboflow)  
+ Custom Negative Samples

**Base Dataset Size:**  
407 positive images (birds) with OBB annotations  

**Negative Samples:**  
~60 images (15% of total) consisting of empty skies, people, indoor scenes, and aircraft to reduce false positives.

**Classes:**  
1 class — *bird*

**Acquisition:**  
Dataset will be downloaded directly in YOLOv8-OBB format via the Roboflow API.  
Negative samples will be obtained from public domain repositories (Pexels, Pixabay) and integrated into the training set using empty label files.

## 5. Technical Approach
**Architecture:**  
YOLO11n-OBB (Nano variant), chosen for its superior accuracy-to-speed ratio and ~22% fewer parameters than YOLOv8.

**Model:**  
Pretrained YOLO11n-OBB weights (COCO) fine-tuned on the custom dataset.

**Framework:**  
PyTorch via the Ultralytics library

**Hardware:**  
Google Colab (T4 GPU) or Local NVIDIA GPU (RTX 3060+)

**Pipeline:**
- **Preprocessing:** Resize to 640×640, normalization, integration of negative samples  
- **Augmentation:** Mosaic (1.0), Mixup (0.5), ±180° rotation (OBB-aware), HSV color jittering  
- **Training:** 150 epochs, batch size 8–16, LR = 0.001, with backbone freezing for first 10 epochs  
- **Evaluation:** mAP@0.5, Precision–Recall curves, and false-positive rate analysis on the negative test set  

## 6. Expected Challenges & Mitigations
**Challenge:** Small Dataset (407 positive images)  
**Mitigation:** Heavy augmentations (Mosaic, Mixup, Rotation) and transfer learning from pretrained YOLO11n models to prevent overfitting.

**Challenge:** High False Positive Rate  
**Mitigation:** Inclusion of 15% negative samples to explicitly teach the model what not to detect, improving robustness.

**Challenge:** Rotated Objects  
**Mitigation:** Use of Oriented Bounding Boxes (OBB) ensures precise localization of birds in arbitrary orientations, improving detection accuracy.
