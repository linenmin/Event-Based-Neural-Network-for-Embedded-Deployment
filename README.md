# Event-Based Neural Network for Embedded Deployment
## Representation-Model Co-Design and Event-Data Acquisition System

This repository contains the implementation of the master's thesis project focusing on event-based vision systems and neural network deployment. The project explores the co-design of event data representations and neural network architectures for efficient embedded deployment.

---

## 📚 Chapter 3: Training and Evaluation of Neural Networks on CIFAR10-DVS

The `Searching_Training_Model` folder contains the implementation of different event data representations and model training approaches for the CIFAR10-DVS dataset:

| File | Description |
|------|-------------|
| `voxel_transformer_64_64_3.ipynb` | Converts event data into 64x64x3 voxel representation |
| `timeStack_tf2_transform_numpyFast.ipynb` | Transforms event data into time-stacked representation and saves as TFRecord format |
| `sae_balance_transform.ipynb` | Processes Surface Active Events (SAE) data with balanced representation |
| `histogram2Channel_transform.ipynb` | Converts event data into 2-channel histogram representation |
| `tf2_mobileNetV2.ipynb` | Implements MobileNetV2-based neural network training for event data classification |

---

## 📚 Chapter 4: Neural Network Deployment Strategies

The `Deployment_VitisAI` folder contains the implementation of model optimization and deployment strategies using Vitis AI:

| File | Description |
|------|-------------|
| `pruning_tf2.py` | Implements iterative model pruning to reduce model size while maintaining accuracy |
| `quantize_tf2.py` | Performs model quantization to convert floating-point model to fixed-point format |
| `inspectTest_tf2.py` | Checks model compatibility with DPU deployment and generates inspection reports |
| `DPU_tf2_mobileNetV2_multithread.py` | Implements multi-threaded DPU inference for optimized model deployment |

---

## 📚 Chapter 5: Event Data Recording Platform System

The `recording_platform` folder contains the implementation of a multi-camera event data acquisition and processing system based on circular motion tracking, enabling synchronized timestamp alignment, tag-assisted ROI tracking, and event data generation for robust neural network evaluation.

| File | Description |
|------|-------------|
| `k_search.ipynb` | Main data processing notebook for multi-camera synchronization and tag detection |
| `k_search_funciton/` | Supporting functions for data loading, processing, and event generation |

Key features:
- Multi-camera data synchronization (DV and PI cameras)
- Timestamp alignment and calibration
- Tag-assisted region of interest (ROI) tracking
- Support for multiple data formats (.npy, .dat, .txt, .csv)
- Data visualization and analysis tools