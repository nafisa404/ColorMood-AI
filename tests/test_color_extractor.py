import numpy as np
import cv2
from src.color_extractor import extract_dominant_colors

def test_extract_dominant_colors():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = [0, 255, 0]
    colors = extract_dominant_colors(img, k=1)
    assert colors.shape == (1, 3)
