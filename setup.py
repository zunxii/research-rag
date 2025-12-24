"""
Setup script for the research package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="medical-multimodal-research",
    version="1.0.0",
    description="Multimodal Medical Image-Text Retrieval with Counterfactual Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Research Team",
    python_requires=">=3.8",
    packages=find_packages(include=["core", "core.*"]),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "open_clip_torch>=2.20.0",
        "peft>=0.4.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "spacy>=3.5.0",
        "scispacy>=0.5.0",
        "google-genai>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "research-cli=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

