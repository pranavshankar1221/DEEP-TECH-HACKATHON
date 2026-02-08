# Dataset Description

## Dataset Name
WM-811K Semiconductor Wafer Map Dataset

## Overview
The WM-811K dataset contains labeled semiconductor wafer map images used for defect pattern classification. Each image represents a wafer or die layout with spatial defect distributions captured during semiconductor manufacturing.

## Purpose
The dataset is used to train and evaluate an Edge-AI model capable of classifying wafer defect patterns in real time on embedded devices, enabling automated quality inspection.

## Defect Classes
The dataset consists of 9 classes:

1. Center – Defects concentrated at the center of the wafer  
2. Donut – Ring-shaped defect patterns  
3. Edge-Loc – Localized defects near wafer edges  
4. Edge-Ring – Circular defects along the wafer boundary  
5. Local – Small, localized defect clusters  
6. Near-Full – Defects covering most of the wafer  
7. Normal – Clean wafers with no defects  
8. Random – Randomly distributed defect points  
9. Scratch – Linear scratch-like defect patterns  

## Data Split
- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

## Preprocessing
- Converted to grayscale
- Resized to 128 × 128
- Normalized pixel values
- Augmented using horizontal flips and minor rotations

## Relevance to Edge AI
- Compact image size suitable for low-memory devices
- Clear defect patterns enable lightweight CNN learning
- Ideal for offline, real-time inference on embedded hardware
