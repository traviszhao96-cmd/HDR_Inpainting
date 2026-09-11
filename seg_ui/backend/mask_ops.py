"""Binary removal-mask cleanup; distances are in original image pixels."""
import cv2
import numpy as np


def fill_holes(mask):
    padded = np.pad(mask, 1)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    return (padded | cv2.bitwise_not(flooded))[1:-1, 1:-1]


def single_component(mask, seed=None):
    binary = (np.asarray(mask) > 127).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    selected = 0
    if seed is not None:
        x, y = seed
        if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1]:
            selected = int(labels[y, x])
    if selected == 0:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == selected).astype(np.uint8) * 255


def refine_mask(mask, inward=0, outward=2, smooth=3, seed=None):
    result = (np.asarray(mask) > 127).astype(np.uint8) * 255
    # Remove islands before dilation can connect them to the target.
    result = fill_holes(single_component(result, seed))
    for radius, operation in ((smooth, cv2.MORPH_CLOSE), (inward, cv2.MORPH_ERODE), (outward, cv2.MORPH_DILATE)):
        radius = max(0, min(32, int(radius)))
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
            result = cv2.morphologyEx(result, operation, kernel)
    if smooth:
        result = (cv2.GaussianBlur(result, (0, 0), max(0.5, smooth / 2)) >= 128).astype(np.uint8) * 255
    return fill_holes(single_component(result, seed))
