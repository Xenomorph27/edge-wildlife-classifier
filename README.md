# Edge Wildlife Classifier

Real-time wildlife species classification (Deer / Elk / Moose) using ResNet18 
fine-tuned in PyTorch, deployed for live webcam inference via OpenCV.

## Results
- **Accuracy:** ~88% on 3-class classification
- **Classes:** Deer, Elk, Moose
- **Architecture:** ResNet18 with custom fully connected head

## Features
- RGB preprocessing pipeline with 224×224 resizing and tensor normalization
- Confidence-aware prediction with class-specific thresholds
- "UNSURE" handling for low-confidence predictions
- CPU/GPU adaptive inference
- Live webcam feed via OpenCV

## Tech Stack
- PyTorch, torchvision
- OpenCV
- ResNet18 (pretrained on ImageNet, fine-tuned)

## Usage
```bash
pip install torch torchvision opencv-python
python Animal_Classifier_Lite/inference.py
```

## Architecture
ResNet18 pretrained backbone → Custom FC layer → Softmax (3 classes)
