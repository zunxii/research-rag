"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Device for testing."""
    return "cpu"


@pytest.fixture(scope="session")
def test_image_path():
    """Path to test image."""
    return Path("data/images/edema_Image_1.jpg")


@pytest.fixture(scope="session")
def test_image(test_image_path):
    """Load test image."""
    if test_image_path.exists():
        return Image.open(test_image_path).convert("RGB")
    else:
        # Create dummy image if not found
        return Image.new("RGB", (224, 224), color="red")


@pytest.fixture(scope="session")
def test_text():
    """Sample clinical text."""
    return "swelling noted around joint swollen ankle with pitting"


@pytest.fixture(scope="session")
def kb_dir():
    """Path to smoke test KB."""
    return Path("outputs/kb/kb_smoke")


@pytest.fixture(scope="session")
def trained_lora_path():
    """Path to trained LoRA."""
    path = Path("outputs/models/trained_lora")
    if path.exists():
        return str(path)
    return None


@pytest.fixture(scope="session")
def trained_fusion_path():
    """Path to trained fusion."""
    path = Path("outputs/models/trained_fusion/fusion.pt")
    if path.exists():
        return str(path)
    return None
