# Model Details

## Model Name
SimpleCNN – Lightweight Convolutional Neural Network

## Objective
To accurately classify semiconductor wafer defect patterns while remaining computationally efficient for edge deployment.

## Architecture Overview
The model consists of:

- Three convolutional blocks:
  - Conv2D → ReLU → MaxPooling
- Increasing feature depth: 32 → 64 → 128 channels
- Fully connected classifier with dropout regularization

## Input Specifications
- Image Size: 128 × 128
- Channels: 1 (Grayscale)

## Output
- 9-class softmax classification corresponding to defect types

## Optimization Strategy
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.0001
- Dropout: 0.3
- Batch Size: 16
- Epochs: 4

## Training Performance (Phase-1)
- Best Validation Accuracy: 95.44%
- Stable convergence with minimal overfitting
- Model saved automatically based on best validation score

## Model Format
- Framework: PyTorch
- Saved Format: `.pth`
- Checkpoint includes:
  - Model weights
  - Class labels
  - Validation accuracy

## Design Rationale
This CNN was intentionally designed to be:
- Lightweight
- Fast during inference
- Easily portable to edge platforms
- Suitable for future conversion to ONNX / TFLite
