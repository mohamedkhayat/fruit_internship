# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys

project = "Fruit Internship"
copyright = "2025, Mohamed Khayat"
author = "Mohamed Khayat"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_mock_imports = [
    "omegaconf",
    "torch",
    "albumentations",
    "numpy",
    "matplotlib",
    "cv2",
    "hydra",
    "tqdm",
    "transformers",
    "pandas",
    "seaborn",
]

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(PROJECT_ROOT))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
