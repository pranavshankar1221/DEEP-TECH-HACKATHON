# Edge-AI Defect Classification for Semiconductor Wafer/Die Images

**Team Name:** WebNovas_PS01  
**Problem Statement Code:** PS01 – Semiconductor Defect Detection  
**Hackathon:** DeepTech Hackathon 2026  
**Phase:** Phase-1 (Model Development & Baseline Validation)

---

## 1. Problem Overview

Semiconductor manufacturing requires highly accurate inspection to detect wafer and die defects that affect yield and reliability. Traditional manual inspection is time-consuming, costly, and prone to human error. This project addresses the problem by developing an automated, edge-AI based defect classification system using deep learning.

---

## 2. Edge-AI Relevance

- Real-time defect detection on manufacturing lines  
- Offline processing without cloud dependency  
- Reduced inspection cost and latency  
- Scalable deployment across multiple edge devices  
- Improved data privacy and security  

---

## 3. Solution Summary

The proposed solution uses a lightweight CNN-based image classification pipeline deployed on edge hardware. Wafer images are preprocessed and passed through a trained deep learning model to classify defects or identify invalid/non-wafer images with confidence scoring.

Pipeline:  
**Image Capture → Preprocessing → CNN Inference → Defect / Invalid Classification → Confidence Output**

---

## 4. Dataset & Class Design

**Dataset:** WM-811K Semiconductor Wafer Dataset  

- **Total Images (planned/current):** ~5,000+  
- **Number of Classes:** 9  
  - 6 defect classes  
  - 1 clean (normal) class  
  - 1 random defect class  
  - 1 other / invalid (non-wafer images)

**Class List:**  
Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Scratch, Random, Normal  

**Train / Validation / Test Split:**  
70% / 15% / 15%  

**Image Type:** Grayscale preferred  
**Labeling Method:** Public dataset with manual verification for invalid samples  

---

## 5. Baseline Model (Phase-1)

- **Architecture:** CNN HYPER-PARAMETER    
- **Input Size:** 224 × 224 × 3  
- **Framework:** PyTorch  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  

---

## 6. Phase-1 Results

| Metric | Value |
|------|------|
| Validation Accuracy | ~95% |
| Training Accuracy | ~94% |
| Model Size | ~21 MB (pre-quantization) |
| Epochs | 10 |

The model demonstrates stable convergence and accurate multi-class classification suitable for edge deployment.

---

## 7. Edge Deployment Readiness

- ONNX conversion supported  
- TensorFlow Lite (TFLite) conversion in progress  

**Target Devices:**  
Raspberry Pi, NXP i.MX series, NVIDIA Jetson Nano/Xavier, industrial edge systems  

---


