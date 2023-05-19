"""
High and low level signals measured within large, uniform region of phantom for
ACR T1 and T2 series slice 7.

Apply circular kernel average filter to the image. Find the smallest and largest
values of the filtered image.

"""
from .image_processing import PreprocessedSlice
from .image_processing import get_circular_mask, pixelate
import numpy as np
import cv2

ROI_LARGE_ACR = 200
ROI_SMALL = 1


class IntensityUniformityTest:
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.large_roi_centre = None
        self.large_roi_radius = None
        self.small_roi_radius = None
        self.high = None
        self.low = None
        self.high_roi_centres = None
        self.low_roi_centres = None

    @property
    def percent_integral_uniformity(self) -> float:
        if self.low is not None and self.high is not None:
            return 100 * (1 - (self.high - self.low) / (self.high + self.low))

    def run(self) -> None:
        x, y, w, h = self.bounding_rectangle
        self.large_roi_centre = pixelate(np.array((y + (h - 1) / 2, x + (w - 1) / 2)))
        small_roi_radius = np.sqrt(ROI_SMALL / np.pi)
        small_roi_radius = pixelate(small_roi_radius * 10 / self.pixel_spacing[0])
        self.small_roi_radius = small_roi_radius

        circle_avg_kernel = np.zeros((2 * small_roi_radius, 2 * small_roi_radius))
        kernel_centre = (small_roi_radius - 1, small_roi_radius - 1)
        kernel_mask = get_circular_mask(
            circle_avg_kernel, kernel_centre, small_roi_radius
        )
        circle_avg_kernel[kernel_mask] = 1
        circle_avg_kernel /= np.sum(circle_avg_kernel)
        filtered_image = cv2.filter2D(self.pixel_array, -1, circle_avg_kernel)

        large_roi_radius = np.sqrt(ROI_LARGE_ACR / np.pi)
        large_roi_radius = pixelate(large_roi_radius * 10 / self.pixel_spacing[0])
        self.large_roi_radius = large_roi_radius
        large_roi_mask = get_circular_mask(
            filtered_image,
            self.large_roi_centre,
            large_roi_radius - 2 * small_roi_radius,
        )
        self.high = filtered_image[large_roi_mask].max()
        self.low = filtered_image[large_roi_mask].min()

        high_inds = np.where(filtered_image[large_roi_mask] == self.high)
        low_inds = np.where(filtered_image[large_roi_mask] == self.low)
        mask_inds = np.where(large_roi_mask)
        self.high_roi_centres = zip(mask_inds[0][high_inds], mask_inds[1][high_inds])
        self.low_roi_centres = zip(mask_inds[0][low_inds], mask_inds[1][low_inds])
