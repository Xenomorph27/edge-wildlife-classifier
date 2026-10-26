<img src="docs/assets/banner.png" alt="Edge Wildlife Classifier — live deer, elk and moose recognition" width="100%">

<p align="center">
  <img alt="ResNet18" src="https://img.shields.io/badge/ResNet18-transfer_learning-84cc16?style=flat-square&labelColor=0f1a08&logo=pytorch&logoColor=white">
  <img alt="accuracy" src="https://img.shields.io/badge/accuracy-~88%25-84cc16?style=flat-square&labelColor=0f1a08">
  <img alt="classes" src="https://img.shields.io/badge/classes-deer_·_elk_·_moose-84cc16?style=flat-square&labelColor=0f1a08">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-live_webcam-84cc16?style=flat-square&labelColor=0f1a08&logo=opencv&logoColor=white">
  <img alt="Raspberry Pi" src="https://img.shields.io/badge/target-Raspberry_Pi-84cc16?style=flat-square&labelColor=0f1a08&logo=raspberrypi&logoColor=white">
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-84cc16?style=flat-square&labelColor=0f1a08"></a>
</p>

```console
$ python Animal_Classifier_Lite/classify.py
Model loaded on cpu
Classes: ['deer', 'elk', 'moose']
```

<p align="center"><sub>Opens the webcam and labels every frame <code>DEER</code>, <code>ELK</code>, <code>MOOSE</code> — or <code>UNSURE</code> when it isn't confident. Press <kbd>q</kbd> to quit.</sub></p>

**Contents:** [Highlights](#highlights) / [How it works](#how-it-works) / [Quick start](#quick-start) / [Train your own](#train-your-own)

## Changelog

```diff
+ 2026-10  runs from any folder · requirements.txt · new docs
+ 2026-04  first release: ResNet18 deer / elk / moose classifier, live webcam, per-class thresholds
```

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
