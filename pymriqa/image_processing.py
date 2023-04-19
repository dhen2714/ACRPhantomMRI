import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessedSlice:
    pixel_array: np.ndarray
    pixel_spacing: tuple[float]
    binary_threshold: float
    binary_image: np.ndarray
    bounding_rectangle: tuple[int]
    water_mean: float


def pixelate(value: float) -> int:
    """Converts floating point value to nearest integer."""
    return np.round(value).astype(int)


def preprocess_slice(
    image: np.ndarray, pixel_spacing: tuple[float], intensity_threshold=0.1
) -> PreprocessedSlice:
    water_mean = phantom_mpv(image, intensity_threshold=intensity_threshold)
    binary_threshold = water_mean / 4
    binary = binary_image(image, threshold=binary_threshold)
    bounding_rect = bounding_rectangle(binary)
    preprocessed = PreprocessedSlice(
        image,
        pixel_spacing,
        binary_threshold,
        binary,
        bounding_rect,
        water_mean,
    )
    return preprocessed


def binary_image(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    image = image > threshold
    return cv2.medianBlur(image.astype(np.uint8), 3)


def phantom_mpv(image: np.ndarray, intensity_threshold: float = 0.1) -> float:
    """
    Get mean pixel value of ACR phantom pixels.
    Pixels with values that are below intensity_threshold * (maximum
    pixel value) are not included in the calculation of the mean.
    """
    pixel_val_threshold = image.max() * intensity_threshold
    mask = image > pixel_val_threshold
    return image[mask].mean()


def bounding_rectangle(binary_image: np.ndarray) -> tuple[int]:
    """
    Returns top left corner column index, row index, width and height of the
    rectangle that bounds the binarised ACR phantom.
    """
    contours, _ = cv2.findContours(
        binary_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    max_len = 0
    for i, c in enumerate(contours):
        contour_length = len(c)
        if contour_length > max_len:
            max_len = contour_length
            contour_ind = i
    phantom = contours[contour_ind]
    x, y, w, h = cv2.boundingRect(phantom)
    return (x, y, w, h)
