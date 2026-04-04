# Edge Wildlife Classifier

A lightweight edge AI system designed to classify wildlife species (deer, elk, moose) using efficient computer vision techniques. Built for deployment on resource-constrained devices like Raspberry Pi.

---

##  Features

- Lightweight and fast classification
- Designed for edge devices (Raspberry Pi)
- Works on real-world wildlife images
- Handles low-to-medium quality inputs
- Modular and extensible pipeline

---

##  System Overview

The system processes cropped animal images and performs classification using:

1. Image preprocessing  
2. Silhouette extraction  
3. Feature extraction  
4. Rule-based / ML classification  

---

##  Project Structure
edge-wildlife-classifier/
│
├── classify.py # Main classification script
├── silhouette.py # Image preprocessing & silhouette extraction
├── test_images/ # Sample images for testing
├── model/ # Trained model (if included)
├── requirements.txt # Dependencies
└── README.md


---

## Installation

```bash
git clone https://github.com/Xenomorph27/edge-wildlife-classifier.git
cd edge-wildlife-classifier
pip install -r requirements.txt
```

## Usage
In Termial navigate to respective folder and  and run:
```bash
python classify.py
```
