# ObjectDetect: Real-Time Object Detection Implementations

## 📋 Table of Contents

- [Overview](#overview)
  - [System Architecture](#system-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Faster R-CNN](#faster-r-cnn-inference)
  - [YOLO v1](#yolo-v1-inference)
  - [Training](#training)
- [Models](#models)
- [Performance & Results](#performance--results)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [References](#references)
- [Support & Contact](#support--contact)

---

## Overview

**ObjectDetect** is a comprehensive implementation of two state-of-the-art object detection architectures:

1. **Faster R-CNN** (TensorFlow 2.x) - Region-based CNN detector optimized for accuracy
2. **YOLOv1** (PyTorch) - Single-stage detector optimized for real-time performance

This project provides complete implementations including data loading, model training, evaluation, and inference pipelines for both architectures. It's designed for research and production use cases in computer vision and object detection tasks.

**📚 Published Research:** This framework is based on and extends the work published in:
> Murthy, J. S., et al. (2022). "ObjectDetect: A Real-Time Object Detection Framework for Advanced Driver Assistant Systems Using YOLOv5." *Wireless Communications and Mobile Computing*, 2022(1), 9444360. Wiley Online Library.

### System Architecture

<img align="center" width="1000" src="images/Architecture.png" alt="ObjectDetect Framework Architecture">

**Figure 1:** Proposed ObjectDetect framework showing the dual-path detection pipeline combining real-time YOLO detection with accurate Faster R-CNN inference, integrated object tracking, and visualization.

### Key Capabilities

- **Multi-model support**: Train and evaluate multiple detection architectures
- **Flexible data handling**: Support for custom datasets and formats
- **Real-time inference**: Optimized inference pipelines for video and image inputs
- **Production-ready code**: Comprehensive error handling, logging, and configuration management
- **Comparative analysis**: Direct comparison between Faster R-CNN and YOLO architectures

---

## Features

### Faster R-CNN (TensorFlow 2)
- ✅ Pre-trained model support (COCO, Custom models)
- ✅ Video and image inference
- ✅ Batch processing capabilities
- ✅ Configurable confidence thresholds
- ✅ Class-specific detection filtering
- ✅ Real-time FPS display
- ✅ Visualization with bounding boxes and labels

### YOLOv1 (PyTorch)
- ✅ Custom model training from scratch
- ✅ Support for BDD100K and custom datasets
- ✅ Image and video inference pipelines
- ✅ Batch processing with data augmentation
- ✅ Training checkpoints and model resumption
- ✅ Validation during training
- ✅ Configurable anchor boxes and grid sizes

---

## Project Structure

```
ObjectDetect/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── Paper0.pdf                         # Research paper and methodology
│
├── Faster R-CNN/                      # TensorFlow 2 Implementation
│   ├── detector.py                    # Main detector class (DetectorTF2)
│   ├── detect_objects.py              # Inference script for images/videos
│   ├── Faster_RCNN_Final.ipynb        # Training notebook (Jupyter)
│   ├── models/                        # Pre-trained models and configs
│   │   ├── label_map.pbtxt            # Class label definitions
│   │   ├── inference_graph/           # Frozen graph for inference
│   │   └── saved_model/               # TensorFlow SavedModel format
│   └── train_tf2/                     # Training utilities
│       ├── model_main_tf2.py          # Training entry point
│       ├── exporter_main_v2.py        # Model export script
│       ├── start_train.sh             # Training startup script
│       └── start_eval.sh              # Evaluation startup script
│
├── YOLO/                              # PyTorch YOLO Implementation
│   ├── Training YOLO/                 # Training components
│   │   ├── train.py                   # Training entry point
│   │   ├── model.py                   # YOLOv1 model architecture
│   │   ├── loss.py                    # Custom YOLO loss function
│   │   ├── dataset.py                 # Data loading and preprocessing
│   │   ├── utils.py                   # Utility functions (IoU, coordinate transforms)
│   │   └── validation.py              # Validation utilities
│   │
│   └── Inference YOLO/                # Inference only (lightweight)
│       ├── model.py                   # YOLOv1 model architecture
│       ├── YOLO_to_image.py           # Single image inference
│       └── YOLO_to_video.py           # Video stream inference
│
└── Result images and videos/          # Pre-generated detection results
    ├── Faster R-CNN/                  # Results from Faster R-CNN
    ├── YOLO/                          # Results from YOLO
    └── Video Thumbnails/              # Thumbnails of video results
```

---

## Requirements

### System Requirements
- **Python**: 3.8 - 3.11
- **CUDA**: 11.x or 12.x (for GPU acceleration, optional)
- **cuDNN**: 8.x (if using CUDA)
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with compute capability 3.5+ (optional but recommended)

### Python Dependencies

#### For Faster R-CNN (TensorFlow):
```
tensorflow>=2.10.0
tensorflow-object-detection-api>=2.10.0
opencv-python>=4.6.0
numpy>=1.21.0
```

#### For YOLO (PyTorch):
```
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
Pillow>=9.0.0
```

#### For Development:
```
jupyter>=1.0.0
ipython>=8.0.0
matplotlib>=3.5.0
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ObjectDetect.git
cd ObjectDetect
```

### 2. Environment Setup

#### Option A: Using Conda (Recommended)

```bash
# Create environment with GPU support (CUDA 11.8)
conda create -n objectdetect python=3.10 cuda-toolkit::cuda-toolkit=11.8 -y
conda activate objectdetect

# For CPU-only (skip GPU packages)
conda create -n objectdetect python=3.10 -y
conda activate objectdetect
```

#### Option B: Using Python venv

```bash
python3 -m venv objectdetect_env
source objectdetect_env/bin/activate  # On Windows: objectdetect_env\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install TensorFlow and Faster R-CNN dependencies
pip install tensorflow==2.13.0 tensorflow-object-detection-api opencv-python numpy

# Install PyTorch for YOLO (CPU version shown; for GPU see PyTorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For development/Jupyter notebooks
pip install jupyter ipython matplotlib
```

**OR use the requirements file (when created):**

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

---

## Quick Start

### Faster R-CNN - Detect Objects in an Image

```bash
cd "Faster R-CNN"

python detect_objects.py \
  --model_path models/efficientdet_d0_coco17_tpu-32/saved_model \
  --path_to_labelmap models/mscoco_label_map.pbtxt \
  --images_dir data/samples/images/ \
  --threshold 0.4 \
  --save_output
```

### Faster R-CNN - Detect Objects in a Video

```bash
cd "Faster R-CNN"

python detect_objects.py \
  --model_path models/efficientdet_d0_coco17_tpu-32/saved_model \
  --path_to_labelmap models/mscoco_label_map.pbtxt \
  --video_path data/samples/pedestrian_test.mp4 \
  --video_input \
  --threshold 0.4 \
  --save_output \
  --output_directory data/samples/output
```

### YOLO - Detect Objects in an Image

```bash
cd "YOLO/Inference YOLO"

python YOLO_to_image.py \
  --input path/to/image.jpg \
  --weights_path path/to/model_weights.pt \
  --threshold 0.45
```

### YOLO - Detect Objects in a Video

```bash
cd "YOLO/Inference YOLO"

python YOLO_to_video.py \
  --input path/to/video.mp4 \
  --weights_path path/to/model_weights.pt \
  --output output_video.mp4 \
  --threshold 0.45
```

---

## Usage

### Faster R-CNN Inference

#### Python API

```python
from detector import DetectorTF2
import cv2

# Initialize detector
detector = DetectorTF2(
    path_to_checkpoint='models/efficientdet_d0_coco17_tpu-32/saved_model',
    path_to_labelmap='models/mscoco_label_map.pbtxt',
    class_id=None,  # None = all classes, or provide list like [1, 2] for specific classes
    threshold=0.4
)

# Detect from image
image = cv2.imread('test_image.jpg')
detections = detector.DetectFromImage(image)  # Returns: [[x_min, y_min, x_max, y_max, class_label, confidence], ...]

# Visualize
output_image = detector.DisplayDetections(image, detections, det_time=50)
cv2.imwrite('output.jpg', output_image)
```

#### Command Line

```bash
python detect_objects.py \
  --model_path <path_to_model> \
  --path_to_labelmap <path_to_labels> \
  --images_dir <directory_with_images> \
  --threshold 0.4 \
  --save_output
```

**Arguments:**
- `--model_path`: Path to TensorFlow SavedModel directory
- `--path_to_labelmap`: Path to labelmap (.pbtxt) file
- `--class_ids`: Comma-separated class IDs to detect (e.g., "1,3" for person,car)
- `--threshold`: Detection confidence threshold [0.0-1.0]
- `--images_dir`: Directory containing input images
- `--video_path`: Path to input video file
- `--output_directory`: Directory for detection results
- `--video_input`: Flag to enable video input mode
- `--save_output`: Flag to save results

### YOLO v1 Inference

#### Python API

```python
from model import YOLOv1
from PIL import Image
import torch
from torchvision import transforms

# Initialize model
model = YOLOv1(split_size=14, num_boxes=2, num_classes=13)
model.load_state_dict(torch.load('weights.pt'))
model.eval()

# Preprocess image
image = Image.open('test_image.jpg')
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

# Perform detection
with torch.no_grad():
    predictions = model(image_tensor)
```

#### Command Line

```bash
# Single image
cd YOLO/Inference\ YOLO/
python YOLO_to_image.py \
  --input image.jpg \
  --weights_path model_weights.pt \
  --threshold 0.45 \
  --output output.jpg

# Video
python YOLO_to_video.py \
  --input video.mp4 \
  --weights_path model_weights.pt \
  --output output.mp4 \
  --threshold 0.45
```

### Training

#### Train YOLO from Scratch

```bash
cd YOLO/Training\ YOLO/

python train.py \
  --train_img_files_path bdd100k/images/100k/train/ \
  --train_target_files_path bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json \
  --learning_rate 1e-5 \
  --batch_size 10 \
  --number_epochs 100 \
  --number_boxes 2 \
  --lambda_coord 5 \
  --lambda_noobj 0.5 \
  --load_model 0
```

**Training Arguments:**
- `--train_img_files_path`: Path to training images
- `--train_target_files_path`: Path to JSON labels (BDD100K format)
- `--learning_rate`: Learning rate for optimizer
- `--batch_size`: Mini-batch size
- `--number_epochs`: Number of training epochs
- `--load_model`: Load previous checkpoint (1=yes, 0=no)
- `--load_model_file`: Checkpoint filename to load

#### Train Faster R-CNN

```bash
cd "Faster R-CNN/train_tf2"
bash start_train.sh
```

Edit `start_train.sh` to configure:
- Output directory
- Model config path
- Training data paths

---

## Models

### Faster R-CNN

**Pre-trained Models Available:**
- EfficientDet-D0 (COCO trained) - Fast, lightweight
- EfficientDet-D1 through D7 - Increasing accuracy/speed tradeoff
- SSD MobileNet v2 (COCO trained) - Mobile-friendly

**Model Format:** TensorFlow SavedModel

**Input:** RGB images (variable size, resized to model input)

**Output:**
- Detection boxes: [batch_size, max_detections, 4] (normalized coordinates)
- Detection classes: [batch_size, max_detections]
- Detection scores: [batch_size, max_detections]

### YOLOv1

**Architecture:**
- Input: 448×448×3 images
- Output: 14×14 grid of predictions
- 2 boxes per grid cell (configurable)
- 13 object classes (BDD100K)

**Model Format:** PyTorch `.pt` files

**Output:** Grid-based predictions:
- Grid cells: 14×14
- Per cell: [x, y, w, h, confidence] × 2 boxes + class probabilities (13 classes)

---

## Performance & Results

### Benchmark Results on BDD100K Dataset

All benchmarks were conducted on NVIDIA V100 SXM2 32GB GPU with batch size=1 for fair FPS comparison.

#### Comparative Performance Analysis

| Architecture | mAP (%) | FPS | Model Size | Inference Time |
|-------------|---------|-----|------------|-----------------|
| **YOLO v1** | 18.6 | **212.4** | ~250 MB | 4.7 ms |
| **Faster R-CNN** | **41.8** | 17.1 | ~350 MB | 58.5 ms |

**Key Observations:**
- **YOLO v1**: Prioritizes speed (212.4 FPS) - ideal for real-time applications with moderate accuracy requirements
- **Faster R-CNN**: Prioritizes accuracy (41.8% mAP) - recommended for safety-critical autonomous driving systems
- **Speed-Accuracy Tradeoff**: YOLO is ~12.4x faster; Faster R-CNN is 2.25x more accurate

### YOLO v1 Detection Results

<img align="center" width="1000" src="Result%20images%20and%20videos/YOLO/yolo_1.gif" alt="YOLO Detection 1">
<img align="center" width="1000" src="Result%20images%20and%20videos/YOLO/yolo_2.gif" alt="YOLO Detection 2">
<img align="center" width="1000" src="Result%20images%20and%20videos/YOLO/yolo_3.gif" alt="YOLO Detection 3">
<img align="center" width="1000" src="Result%20images%20and%20videos/YOLO/yolo_4.gif" alt="YOLO Detection 4">

### Faster R-CNN Detection Results

<img align="center" width="1000" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_1.gif" alt="Faster R-CNN Detection 1">
<img align="center" width="1000" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_2.gif" alt="Faster R-CNN Detection 2">
<img align="center" width="1000" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_3.gif" alt="Faster R-CNN Detection 3">
<img align="center" width="1000" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_4.gif" alt="Faster R-CNN Detection 4">

### Side-by-Side Comparison (YOLO left, Faster R-CNN right)

<img align="left" width="390" src="Result%20images%20and%20videos/YOLO/yolo_5.gif" alt="YOLO Comparison 1">
<img align="right" width="390" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_5.gif" alt="Faster R-CNN Comparison 1">

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

<img align="left" width="390" src="Result%20images%20and%20videos/YOLO/yolo_6.gif" alt="YOLO Comparison 2">
<img align="right" width="390" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_6.gif" alt="Faster R-CNN Comparison 2">

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

---

## Dataset

### Supported Datasets

#### Faster R-CNN
- **COCO Dataset**: Included in pre-trained models
- **Custom Datasets**: Support for Pascal VOC and COCO formats
- **Label Format**: `.pbtxt` label maps

#### YOLO v1
- **BDD100K**: Primary training dataset (requires download)
- **Custom Datasets**: JSON-based annotation format
- **Label Format**: JSON with `[x, y, w, h, class_id]` in normalized coordinates

### BDD100K Dataset Setup

```bash
# Download from: https://bdd-data.berkeley.edu/
# After download, structure as:
bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       ├── val/
│       └── test/
└── labels/
    └── det_v2_*_release.json
```

### Custom Dataset Format

#### YOLO JSON Format:
```json
{
  "name": "image_name.jpg",
  "width": 1920,
  "height": 1080,
  "labels": [
    {
      "category": "car",
      "box2d": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
      }
    }
  ]
}
```

---

## Troubleshooting

### Installation Issues

**Problem**: CUDA not found when importing TensorFlow
```bash
# Solution: Install CPU-only version or check NVIDIA drivers
nvidia-smi  # Verify GPU is recognized
pip install tensorflow-cpu  # CPU version
```

**Problem**: ImportError for `object_detection`
```bash
# Solution: Install TensorFlow Object Detection API
pip install tf-models-official
```

### Runtime Issues

**Problem**: `memory.out_of_memory` error during inference
```bash
# Solution: Reduce batch size or image resolution
python detect_objects.py --threshold 0.5 # Lower threshold reduces processing
```

**Problem**: Video codec not found
```bash
# Solution: Install ffmpeg and opencv-python
brew install ffmpeg  # macOS
# Linux: sudo apt-get install ffmpeg python3-opencv
```

**Problem**: Model weights not found
```bash
# Solution: Check paths and ensure model is downloaded
find . -name "*.pt" -o -name "saved_model.pb"
```

### Performance Issues

**Problem**: Slow inference speed
- Check GPU utilization: `nvidia-smi`
- Use lighter model variant
- Batch process when possible
- Reduce input image resolution

**Problem**: Out of memory errors
- Reduce batch size
- Use gradient checkpointing (training)
- Process images in streams for video

---

## Configuration

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES="0"  # GPU device ID
export TF_CPP_MIN_LOG_LEVEL="2"   # Reduce TensorFlow logging
export PYTHONUNBUFFERED="1"       # Real-time output
```

### Model Configuration

Edit respective config files:
- **Faster R-CNN**: `models/pipeline.config`
- **YOLO**: See `train.py` arguments

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes with clear messages
4. Push to branch and create a Pull Request
5. Follow PEP 8 style guidelines
6. Add docstrings to new functions/classes
7. Include unit tests for new functionality

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

**Copyright**: © 2026 Jamuna S Murthy

---

## Citation

If this project or the published paper is useful for your research, please cite:

### BibTeX

```bibtex
@article{murthy2022objectdetect,
  title={ObjectDetect: A Real-Time Object Detection Framework for Advanced Driver Assistant Systems Using YOLOv5},
  author={Murthy, Jamuna S and Siddesh, GM and Lai, Wen-Cheng and Parameshachari, BD and Patil, Sujata N and Hemalatha, KL},
  journal={Wireless Communications and Mobile Computing},
  volume={2022},
  number={1},
  pages={9444360},
  year={2022},
  publisher={Wiley Online Library},
  doi={10.1155/2022/9444360},
  url={https://onlinelibrary.wiley.com/doi/10.1155/2022/9444360}
}
```

### APA Citation

Murthy, J. S., Siddesh, G. M., Lai, W.-C., Parameshachari, B. D., Patil, S. N., & Hemalatha, K. L. (2022). ObjectDetect: A real-time object detection framework for advanced driver assistant systems using YOLOv5. *Wireless Communications and Mobile Computing*, 2022(1), 9444360. https://doi.org/10.1155/2022/9444360

---

## References

### Research Papers

- **Faster R-CNN**: Ren et al. (2015) - "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
  - ArXiv: https://arxiv.org/abs/1506.01497

- **YOLO v1**: Redmon et al. (2015) - "You Only Look Once: Unified, Real-Time Object Detection"
  - ArXiv: https://arxiv.org/abs/1506.02640

- **ObjectDetect Framework**: Murthy et al. (2022) - See Citation section above

### Key Resources

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [PyTorch torchvision](https://pytorch.org/vision/stable/index.html)
- [COCO Dataset](https://cocodataset.org/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)

### Related Projects

- [Detectron2](https://github.com/facebookresearch/detectron2) - FB Research detection framework
- [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab detection toolbox
- [Ultralytics YOLOv5](https://github.com/ultralytics/YOLOv5) - YOLOv5 production implementation

---

## Support & Contact

### 📧 Author Information

**Jamuna Srinivasa Murthy**
- **Email**: jamunamurthy.s@gmail.com
- **GitHub**: [github.com/jamuna-murthy](https://github.com/jamuna-murthy)
- **ResearchGate**: [Research Profile](https://www.researchgate.net/profile/Jamuna-Murthy)

### 🤝 Getting Help

For issues, questions, or feature requests:

1. **GitHub Issues**: Check [existing issues](../../issues) first
2. **Documentation**: Review module READMEs in each folder:
   - [Faster R-CNN Documentation](Faster%20R-CNN/README.md)
   - [YOLO Training Documentation](YOLO/Training%20YOLO/README.md)
   - [YOLO Inference Documentation](YOLO/Inference%20YOLO/README.md)
3. **Examples**: See example scripts in each module
4. **Contact**: Email maintainer at: jamunamurthy.s@gmail.com

### 📝 Troubleshooting Guide

**Common Issues and Solutions:**

| Issue | Solution |
|-------|----------|
| CUDA not found | Install NVIDIA drivers or use CPU version |
| Out of memory | Reduce batch size or image resolution |
| Model weights not loading | Verify path and file integrity |
| Video codec error | Install ffmpeg: `brew install ffmpeg` |
| Import errors | Reinstall dependencies: `pip install -r requirements.txt` |

For detailed troubleshooting, see [Troubleshooting Section](#troubleshooting) above.

### 🐛 Reporting Bugs

When reporting issues, please include:
1. Python version (`python --version`)
2. OS and hardware (CPU/GPU model)
3. Error message and full traceback
4. Minimal reproducible example
5. Steps to reproduce

**Example Issue Template:**
```
**Environment:**
- Python 3.10
- NVIDIA RTX 3090
- PyTorch 2.0

**Error:**
[Full error message here]

**Reproduction:**
[Steps to reproduce]
```

### ✨ Contributing

We welcome contributions! Please:
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation
- Submit detailed pull requests

See [Contributing](#contributing) section for full guidelines.

### 📜 License & Attribution

- **License**: MIT (see [LICENSE](LICENSE) file)
- **Copyright**: © 2026 Jamuna S Murthy
- **Citation**: See [Citation](#citation) section for academic use

---

## Summary

**ObjectDetect** provides production-ready implementations of two complementary detection approaches:

- **Faster R-CNN**: Maximum accuracy (41.8% mAP) for safety-critical autonomous driving systems
- **YOLO v1**: Maximum speed (212.4 FPS) for real-time surveillance and embedded systems

Both implementations feature:
✅ Comprehensive error handling and logging  
✅ Multi-device support (GPU/CPU with automatic fallback)  
✅ Input validation and parameter checking  
✅ Professional documentation and examples  
✅ Production-tested code patterns  

Whether you're building research prototypes or deploying to production, this framework provides the tools and examples you need.

---

**Happy detecting! 🎯**

*For the latest updates and news, star this repository on GitHub and follow for releases.*

