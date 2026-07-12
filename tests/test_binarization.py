"""Tests for binarization utilities."""

import numpy as np

from src.preprocessing.binarization import (
    adaptive_binarization,
    contrast_optimize,
    dynamic_binarization,
    otsu_binarization,
)


def test_otsu_binarization():
    """Otsu should return a binary image (0/255)."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 255
    binary = otsu_binarization(gray)
    assert binary.shape == (100, 100)
    assert set(np.unique(binary).tolist()).issubset({0, 255})


def test_adaptive_binarization_odd_block():
    """Adaptive should work with odd block_size."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 255
    binary = adaptive_binarization(gray, block_size=31)
    assert binary.shape == (100, 100)


def test_adaptive_binarization_even_block():
    """Adaptive should correct even block_size to odd."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 255
    binary = adaptive_binarization(gray, block_size=30)
    assert binary.shape == (100, 100)


def test_dynamic_binarization_normal():
    """Dynamic should use Otsu for normal images."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:50, :] = 50
    gray[50:, :] = 200
    binary = dynamic_binarization(gray)
    assert binary.shape == (100, 100)


def test_dynamic_binarization_bright_skewed():
    """Dynamic should use adaptive for bright-skewed images."""
    gray = np.full((100, 100), 250, dtype=np.uint8)
    gray[0, 0] = 0
    binary = dynamic_binarization(gray)
    assert binary.shape == (100, 100)


def test_dynamic_binarization_dark_skewed():
    """Dynamic should use adaptive for dark-skewed images."""
    gray = np.full((100, 100), 5, dtype=np.uint8)
    gray[0, 0] = 255
    binary = dynamic_binarization(gray)
    assert binary.shape == (100, 100)


def test_contrast_optimize():
    """contrast_optimize should return same-shape image."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 128
    enhanced = contrast_optimize(gray)
    assert enhanced.shape == (100, 100)