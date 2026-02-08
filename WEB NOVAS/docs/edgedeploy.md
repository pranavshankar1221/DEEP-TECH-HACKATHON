# Edge Deployment Strategy

## Deployment Goal
To run semiconductor defect classification directly on edge devices without relying on cloud infrastructure, ensuring low latency and high reliability.

## Target Devices
- Raspberry Pi
- ARM-based embedded boards
- Industrial edge gateways
- Low-power inspection systems

## Edge-AI Advantages
- Offline inference capability
- Reduced latency in production lines
- Lower operational costs
- Improved data privacy

## Inference Pipeline
1. Image capture from camera or local storage
2. Preprocessing (resize, grayscale, normalization)
3. CNN inference using PyTorch
4. Confidence and entropy-based validation
5. Output classification or rejection of invalid images

## Invalid Image Handling
To prevent false predictions:
- Confidence thresholding is applied
- Entropy-based uncertainty detection is used
- Non-wafer images (e.g., animals, objects) are rejected

## Optimization Readiness
- Model is lightweight and quantization-ready
- Can be converted to ONNX or TensorFlow Lite
- Suitable for INT8 optimization in Phase-2

## Phase-1 Status
- Model trained and validated
- Edge inference logic implemented
- Conversion pipeline prepared

## Phase-2 Plan
- Model quantization
- Hardware benchmarking
- Real-time camera integration
- Production-level deployment
