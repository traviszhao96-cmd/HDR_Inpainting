"""SDR-only compositing; generation context is not the final edit footprint."""
import cv2
import numpy as np


def expand_generation_mask(mask, radius=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def composite_sdr(source, generated, mask, feather=6):
    """Match low-frequency boundary color, then blend strictly inside the mask.

    Use only known pixels outside the final mask for color estimation, never
    the removed subject. Limit correction to avoid changing scene lighting.
    This cannot repair mismatched geometry or shadows outside the selection.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return source.copy()
    src = source.astype(np.float32)
    gen = generated.astype(np.float32)
    ring = expand_generation_mask(mask, 8) & ~mask
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if ring.sum() >= 32:
        # Smooth residuals rather than the image: retain generated texture.
        residual = np.clip(src - gen, -32, 32) * ring[:, :, None]
        weights = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 16)
        delta = cv2.GaussianBlur(residual, (0, 0), 16)
        delta /= np.maximum(weights[:, :, None], 1e-5)
        delta[weights < 0.005] = 0
        gen += np.clip(delta, -16, 16) * np.exp(-distance[:, :, None] / 24)
    t = np.clip((distance - 1) / max(feather, 1), 0, 1)
    alpha = (t * t * (3 - 2 * t))[:, :, None]
    out = np.rint(np.clip(src * (1 - alpha) + gen * alpha, 0, 255)).astype(np.uint8)
    out[~mask] = source[~mask]
    return out
