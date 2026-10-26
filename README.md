<div align="center">

<img src="docs/assets/banner.png" alt="Edge Wildlife Classifier" height="72">

<br/>

![Accuracy](https://img.shields.io/badge/Accuracy-~88%25-84CC16?labelColor=30363d) ![Classes](https://img.shields.io/badge/Classes-deer_·_elk_·_moose-84CC16?labelColor=30363d) ![ResNet18](https://img.shields.io/badge/ResNet18-transfer_learning-EE4C2C?labelColor=30363d&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-live_webcam-5C3EE8?labelColor=30363d&logo=opencv&logoColor=white) ![Edge](https://img.shields.io/badge/Edge-Raspberry_Pi-A22846?labelColor=30363d&logo=raspberrypi&logoColor=white) [![License](https://img.shields.io/badge/License-MIT-84CC16?labelColor=30363d)](LICENSE)

<br/>

<img src="docs/assets/card.png" alt="Real-time on the edge — ~88% on deer, elk and moose" height="60">

<br/><br/>

[Highlights](#highlights) | [How it works](#how-it-works) | [Quick start](#quick-start) | [Train your own](#train-your-own)

</div>

---

# Edge Wildlife Classifier: Real-Time Deer, Elk and Moose Recognition

A ResNet18 fine-tuned in PyTorch that classifies wildlife live from a camera feed, with per-class confidence thresholds so it says **"UNSURE"** instead of guessing — small enough for edge hardware.

## News

- [2026-10] Runs from any folder, with a `requirements.txt` and refreshed documentation.
- [2026-04] **Released:** ResNet18 deer / elk / moose classifier with live webcam inference and per-class confidence thresholds.


---

## Highlights

- **~88% accuracy** on three visually similar species — deer, elk and moose
- **Real time** — reads the webcam, classifies every frame, overlays the label and confidence
- **Honest when unsure** — each class has its own threshold (elk 75%, deer and moose 60%); anything below shows `UNSURE`
- **Small** — a 45 MB ResNet18 checkpoint that runs on CPU, and uses CUDA automatically when present

## How it works

```mermaid
flowchart LR
  A[🎥 Webcam frame] --> B[BGR → RGB<br/>resize 224×224<br/>to tensor]
  B --> C[ResNet18 backbone<br/>ImageNet-pretrained]
  C --> D[Custom FC head<br/>3 classes]
  D --> E[Softmax]
  E --> F{confidence ≥<br/>class threshold?}
  F -->|yes| G[DEER / ELK / MOOSE]
  F -->|no| H[UNSURE]
```

Deer, elk and moose share colour, posture and habitat, so each class gets its own threshold and elk
the strictest. Showing `UNSURE` costs far less than a confident wrong answer
when a classifier like this feeds a wildlife counter or a road-crossing alert.

## Quick start

The checkpoint is stored with Git LFS, so install [Git LFS](https://git-lfs.com) before cloning.

```bash
git lfs install
git clone https://github.com/Xenomorph27/edge-wildlife-classifier.git
cd edge-wildlife-classifier
pip install -r requirements.txt

python Animal_Classifier_Lite/classify.py     # opens the webcam — press q to quit
```

## Train your own

`train_classifier.py` fine-tunes ResNet18 on any folder-per-class dataset:

```
dataset_final/
├── deer/   *.jpg
├── elk/    *.jpg
└── moose/  *.jpg
```

```bash
cd Animal_Classifier_Lite
python train_classifier.py     # writes animal_classifier.pth (weights + class names)
```

| Setting | Value |
|---|---|
| Backbone | ResNet18, ImageNet weights, new fully connected head |
| Split | 80% train / 20% validation |
| Augmentation | Random horizontal flip, brightness/contrast jitter |
| Optimizer | Adam, lr 3e-4, batch 32, 10 epochs |

Class names are saved inside the checkpoint, so `classify.py` adapts to any number of classes.

## Project structure

```
Animal_Classifier_Lite/
├── classify.py             # live webcam inference
├── train_classifier.py     # fine-tuning script
└── animal_classifier.pth   # trained weights (Git LFS)
```

## Related

[**wildlife-classifier-prov2**](https://github.com/Xenomorph27/wildlife-classifier-prov2) — the successor:
YOLOv8x detection + EfficientNet-B4 classification across 52 species.

## License

[MIT](LICENSE) © Barnak Roy Nandy
