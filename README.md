# weed_crop_detection
# YOLOv8x Weed-Crop Detection for Agricultural Applications

A state-of-the-art computer vision solution for automated crop and weed detection using YOLOv8x deep learning model. This project aims to help agricultural sector with precision farming by accurately identifying and distinguishing between crops and weeds in field images.

## 🌾 Project Overview

This project implements an advanced object detection system specifically designed for agricultural applications. Using the latest YOLOv8x architecture, the model can efficiently detect and classify crops and weeds in various field conditions, enabling farmers to make informed decisions about targeted herbicide application and crop management.

## 📊 Datasets

The model is trained on a comprehensive dataset combining two high-quality agricultural datasets:

### 1. WeedCrop Image Dataset
- **Source**: [Kaggle - WeedCrop Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset)
- **Size**: 2,822 images
- **Quality**: Various resolutions and field conditions
- **Format**: YOLO format with bounding box annotations

### 2. LincolnBeet Dataset
- **Source**: [ActiveLoop - LincolnBeet Dataset](https://datasets.activeloop.ai/docs/ml/datasets/lincolnbeet-dataset/#lincoinbeet-dataset)
- **Size**: 4,402 high-resolution images
- **Resolution**: 1920 x 1080 pixels
- **Quality**: High-quality field images with precise annotations

### Combined Dataset Statistics
- **Total Images**: 7,224
- **Train Set**: 5,558 images (77%)
- **Validation Set**: 676 images (9.4%)
- **Test Set**: 990 images (13.6%)
- **Classes**: 2 (crop, weed)

## 🏗️ Model Architecture

**YOLOv8x** - The largest and most accurate variant of the YOLOv8 family
- **Framework**: Ultralytics YOLOv8
- **Model Size**: YOLOv8x (extra-large)
- **Input Resolution**: 640x640 pixels
- **Classes**: Binary classification (crop vs weed)

## 🚀 Training Configuration

```python
model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    optimizer='AdamW',
    lr0=1e-3,
    batch=16,
    device='cuda',
    seed=69
)
```

### Training Parameters
- **Epochs**: 50
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Batch Size**: 16
- **Image Size**: 640x640
- **Hardware**: GPU (CUDA)

## 📈 Performance Metrics

### Model Performance
- **mAP@0.5**: 0.796 (79.6%)
- **mAP@0.5:0.95**: 0.573 (57.3%)
- **mAP@0.75**: 0.630 (63.0%)
- **Precision**: 0.811 (81.1%)
- **Recall**: 0.753 (75.3%)
- **F1-Score**: 0.781 (78.1%)

### Key Findings
- **Crop Detection**: High precision and recall for crop identification
- **Weed Detection**: Moderate performance, requires additional training data
- **Main Challenge**: Model occasionally misclassifies objects as background
- **False Negatives**: 903 weeds misclassified as background
- **False Positives**: 955 background objects misclassified as weeds

## 📁 Project Structure

```
weed-crop-detection/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── models/
│   └── best.pt
├── results/
│   ├── confusion_matrix.png
│   ├── F1_curve.png
│   ├── PR_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   └── results.csv
├── dataset.yaml
├── yolov8x-weed-crop-detection.ipynb
└── README.md
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install pillow
pip install pyyaml
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd weed-crop-detection

# Install dependencies
pip install -r requirements.txt

# Download datasets (follow dataset links above)
# Organize data according to project structure

# Run training
python train.py

# Run inference
python detect.py --source path/to/image --weights models/best.pt
```

## 💡 Usage Examples

### Training a New Model
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8x.pt')

# Train on custom dataset
model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

### Running Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Batch Processing
```python
# Process multiple images
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

for r in results:
    r.show()  # Display results
    r.save()  # Save annotated images
```

## 📊 Evaluation Results

### Performance Analysis
The model demonstrates strong performance for crop detection but shows room for improvement in weed identification. The precision-recall curves indicate:

- **Crops**: Excellent detection capability with high confidence
- **Weeds**: Moderate performance requiring additional training data
- **Overall**: Suitable for practical agricultural applications with continued improvement

### Confusion Matrix Analysis
- Most classification errors involve background misclassification
- Need for more diverse weed samples in training data
- Balanced performance across both classes with slight bias toward crops

## 🔄 Future Improvements

### Recommended Enhancements
1. **Data Augmentation**: Increase dataset diversity with more labeled weed samples
2. **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and architectural parameters
3. **Multi-Scale Training**: Implement different input resolutions for robustness
4. **Ensemble Methods**: Combine multiple models for improved accuracy
5. **Real-time Optimization**: Model quantization for edge deployment
6. **Multi-Spectral Data**: Incorporate NIR and other spectral bands

### Technical Improvements
- **Advanced Augmentation**: Implement CutMix, MixUp, and Mosaic augmentation
- **Loss Function Optimization**: Experiment with focal loss and other advanced loss functions
- **Architecture Variants**: Test YOLOv9, YOLOv10, or other recent architectures
- **Transfer Learning**: Fine-tune from domain-specific pre-trained models

## 🌐 Applications

### Agricultural Use Cases
- **Precision Herbicide Application**: Targeted weed control reducing chemical usage
- **Crop Monitoring**: Automated field surveillance and health assessment
- **Yield Prediction**: Early-stage crop counting and density estimation
- **Robotic Farming**: Integration with autonomous agricultural machinery
- **Research**: Agricultural research and phenotyping applications

### Commercial Deployment
- **Farm Management Software**: Integration with existing farm management systems
- **Drone Surveillance**: Aerial crop monitoring and mapping
- **Mobile Applications**: Field-ready smartphone applications for farmers
- **IoT Integration**: Edge computing solutions for real-time field monitoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **WeedCrop Dataset** contributors
- **LincolnBeet Dataset** research team
- **Agricultural AI Research Community**

## 📚 References

1. Ultralytics YOLOv8 Documentation
2. WeedCrop Image Dataset - Kaggle
3. LincolnBeet Dataset - ActiveLoop
4. "Agricultural Object Detection: A Survey" - Recent research papers
5. Precision Agriculture and Computer Vision - Related publications

---

**Note**: This project is designed for research and educational purposes. For commercial agricultural applications, please ensure proper validation and testing in your specific field conditions.
