# Fruit Internship Project

Transformer-based object detection pipeline for fruit localization, built as part of a university internship. The current version supports object detection only and is designed to be extended into a multi-task learning setup (e.g., for weight and volume prediction).

---

## Overview

This repository provides a modular and reproducible deep learning pipeline for object detection using transformer models from Hugging Face. It is built with scalability and maintainability in mind, following best practices in code structure, configuration, and logging.

Key components:
- Object detection using RT-DETR v2
- Configurable training via Hydra
- Advanced augmentations with Albumentations
- Experiment tracking with Weights & Biases
- Early stopping and mixed precision training
- Class-imbalanced sampling strategies
- Sphinx-based auto-generated documentation

---

## Directory Structure

```

mohamedkhayat-fruit_internship/
├── conf/                   # Hydra configuration files
│   ├── config.yaml         # Base training config
│   └── model/              # Model-specific configs
├── docs/                   # Sphinx documentation (auto-generated)
├── src/fruit\_project/      # Core codebase (models, training, utils)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Build and tool configuration
└── README.md               # Project description

````

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mohamedkhayat0/fruit_internship.git
cd fruit_internship

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install core dependencies
pip install -e .
pip install -r requirements.txt
````

---

## Usage

### Train with Default Settings

```bash
python src/fruit_project/main.py
```

### Customize with Hydra CLI

```bash
python src/fruit_project/main.py model=detrv2_50 lr=5e-5 aug=safe
```

### Dataset Setup

* Data must be in YOLO format under `data/Fruits-detection/`
* To auto-download from Kaggle, set `download_data: true` in `conf/config.yaml` or `download_data=True` as an argument and set the corresponding environement variables

---

## Features

* Transformer-based object detection (RT-DETRv2)
* Modular model factory (configurable via YAML)
* Safe/strong augmentation pipelines (Albumentations)
* Stratified sampling (max/mean) to handle class imbalance
* Full integration with Weights & Biases:

  * Training/validation metrics
  * Bounding box visualizations
  * Class distribution and confusion matrix plots
  * Checkpoint artifact logging

* Mixed precision support using `torch.cuda.amp`
* Early stopping with automatic best model restoration
* Clean and extensible codebase

---

## Documentation

Auto-generated using Sphinx + AutoAPI.

view docs:
`https://mohamedkhayat.github.io/fruit_internship`

---

## Roadmap

* Support multi-task training (detection + regression)
* Unit test coverage and CI for training and data loaders

---

## License

MIT License © 2025 Mohamed Khayat
