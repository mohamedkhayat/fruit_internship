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

This project supports both a local and a Docker-based setup. Choose the one that best fits your needs.

### 1. Local Setup (Without Docker)

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/mohamedkhayat/fruit_internship.git
cd fruit_internship
```

It is highly recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### For Users (Training & Inference)
If you only want to use the project to train models or run inference, install the core package with the `torch` dependencies:
```bash
pip install .[torch]
```

#### For Developers (Contributing)
If you plan to contribute to the project, you will need the development dependencies (like `ruff`, `mypy`, etc.) in addition to the core package:
```bash
pip install .[dev,torch]
```

### 2. Docker Setup

The recommended way to work with this project is by using a containerized environment.

#### With VS Code (Recommended)
This is the easiest method, ensuring a consistent and reproducible environment.

1.  **Prerequisites:**
    *   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
    *   [Visual Studio Code](https://code.visualstudio.com/)
    *   [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

2.  **Launch:**
    *   Clone the repository.
    *   Open the project folder in VS Code.
    *   A notification will appear asking if you want to "Reopen in Container". Click it.

VS Code will automatically build the Docker image, install all necessary dependencies, and connect to the container. The environment will be ready for both using and developing the project.

#### Without VS Code (Manual Docker Build)
If you are not using VS Code, you can build and run the Docker container manually.

1.  **Prerequisites:**
    *   [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2.  **Build the image:**
    From the project root directory, run:
    ```bash
    docker build -f .devcontainer/Dockerfile -t fruit-internship .
    ```

3.  **Run the container:**
    This command starts an interactive session inside the container, mounts your local project directory, and enables GPU access.

    *On Windows (Command Prompt/PowerShell):*
    ```bash
    docker run --gpus all --ipc=host -v "%cd%:/workspace" -it fruit-internship bash
    ```

    *On macOS/Linux:*
    ```bash
    docker run --gpus all --ipc=host -v "$(pwd):/workspace" -it fruit-internship bash
    ```

4.  **Install dependencies inside the container:**
    Once you have a shell inside the container, install the project dependencies:
    ```bash
    pip install --upgrade pip && pip install -e '.[dev]'
    ```
The environment is now ready.


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
