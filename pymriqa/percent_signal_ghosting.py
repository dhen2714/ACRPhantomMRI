"""
Assesses level of ghosting in slice 7 of ACR T1 images.
"""
from .image_processing import PreprocessedSlice
from .image_processing import pixelate, get_circular_mask
import numpy as np
from dataclasses import dataclass

ROI_LARGE_ACR = 200
ROI_SMALL = 2.5
ROI_SIDE_RATIO = 4


@dataclass
class RectangularROI:
    centre: tuple[int, int]
    width: int
    height: int

    @property
    def corner_index(self) -> tuple[int, int]:
        """Row, column pixel index of the top left corner of the ROI."""
        return (
            self.centre[0] - pixelate(self.height / 2),
            self.centre[1] - pixelate(self.width / 2),
        )

    @property
    def rbounds(self) -> tuple[int, int]:
        """First and last row indices of the ROI."""
        return (self.corner_index[0], self.corner_index[0] + self.height)

    @property
    def cbounds(self) -> tuple[int, int]:
        """First and last column indices of the ROI."""
        return (self.corner_index[1], self.corner_index[1] + self.width)


def select_roi(image: np.ndarray, roi: RectangularROI) -> np.ndarray:
    return image[roi.rbounds[0] : roi.rbounds[1], roi.cbounds[0] : roi.cbounds[1]]


class SignalGhostingTest:
    def __init__(
        self, preprocessed_slice: PreprocessedSlice, large_roi_mask: np.ndarray = None
    ) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.large_roi_mask = large_roi_mask
        self.large_roi_centre = None
        self.large_roi_radius = None
        self.top_roi = None
        self.bottom_roi = None
        self.left_roi = None
        self.right_roi = None
        self.ghosting_ratio = None

    @property
    def percent_ghosting(self) -> float:
        if self.ghosting_ratio is not None:
            return 100 * self.ghosting_ratio

    def calculate_large_roi_mask(self) -> None:
        x, y, w, h = self.bounding_rectangle
        self.large_roi_centre = pixelate(np.array((y + (h - 1) / 2, x + (w - 1) / 2)))
        large_roi_radius = np.sqrt(ROI_LARGE_ACR / np.pi)
        large_roi_radius = pixelate(large_roi_radius * 10 / self.pixel_spacing[0])
        self.large_roi_radius = large_roi_radius
        self.large_roi_mask = get_circular_mask(
            self.pixel_array,
            self.large_roi_centre,
            large_roi_radius,
        )

    def roi_side_lengths(self) -> tuple[int, int]:
        short_length = np.sqrt(ROI_SMALL / ROI_SIDE_RATIO)
        long_length = ROI_SIDE_RATIO * short_length
        short_length_pixels = pixelate(10 * short_length / self.pixel_spacing[0])
        long_length_pixels = pixelate(10 * long_length / self.pixel_spacing[0])
        return short_length_pixels, long_length_pixels

    def draw_rois(self) -> None:
        x, y, w, h = self.bounding_rectangle
        centre_left = (y + pixelate((h - 1) / 2), x)
        centre_top = (y, x + pixelate((w - 1) / 2))
        centre_right = (y + pixelate((h - 1) / 2), x + w - 1)
        centre_bottom = (y + h - 1, x + pixelate((w - 1) / 2))
        # Get distance between each bounding box boundary and the image edge in pixels
        numrows, numcols = self.pixel_array.shape
        left_width = x
        right_width = numcols - centre_right[1]
        top_height = y
        bottom_height = numrows - centre_bottom[0]
        # ROI width to be smaller than the border between bounding box and image edge
        vertical_border_width = np.min((left_width, right_width))
        horizontal_border_height = np.min((top_height, bottom_height))

        # Row, column pixel indices of the centres of each ROI
        roi_top_centre = (
            centre_top[0] - pixelate(horizontal_border_height / 2),
            centre_top[1],
        )
        roi_bottom_centre = (
            centre_bottom[0] + pixelate(horizontal_border_height / 2),
            centre_bottom[1],
        )
        roi_left_centre = (
            centre_left[0],
            centre_left[1] - pixelate(vertical_border_width / 2),
        )
        roi_right_centre = (
            centre_right[0],
            centre_right[1] + pixelate(vertical_border_width / 2),
        )

        short_length, long_length = self.roi_side_lengths()

        self.top_roi = RectangularROI(roi_top_centre, long_length, short_length)
        self.bottom_roi = RectangularROI(roi_bottom_centre, long_length, short_length)
        self.left_roi = RectangularROI(roi_left_centre, short_length, long_length)
        self.right_roi = RectangularROI(roi_right_centre, short_length, long_length)

    def run(self) -> None:
        self.draw_rois()
        top = select_roi(self.pixel_array, self.top_roi).mean()
        bottom = select_roi(self.pixel_array, self.bottom_roi).mean()
        left = select_roi(self.pixel_array, self.left_roi).mean()
        right = select_roi(self.pixel_array, self.right_roi).mean()
        self.calculate_large_roi_mask()
        large_roi_mean = self.pixel_array[self.large_roi_mask].mean()
        self.ghosting_ratio = np.abs(
            ((top + bottom) - (left + right)) / (2 * large_roi_mean)
        )
