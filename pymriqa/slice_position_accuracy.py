"""
Find bar length differences in slices 1 and 11 for ACR T1 and T2 series.
"""
import numpy as np
from .image_processing import PreprocessedSlice
from .image_processing import pixelate


class SlicePositionTest:
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.insert_centre = None
        self.left_column_idx = None
        self.right_column_idx = None
        self.length_left = None
        self.length_right = None
        self.pixel_difference = None

    @property
    def length_difference(self) -> float:
        if self.pixel_difference:
            return self.pixel_difference * self.pixel_spacing[0]

    def preprocess(self) -> None:
        x, y, w, h = self.bounding_rectangle
        # Assume centre of insert is approximately 1/11 down the phantom
        insert_centre = y + pixelate((h - 1) / 11), x + pixelate((w - 1) / 2)
        left_edge_idx = np.where(
            self.binary_image[insert_centre[0], : insert_centre[1]]
        )[0][-1]
        right_edge_idx = (
            np.where(self.binary_image[insert_centre[0], insert_centre[1] :])[0][0]
            + insert_centre[1]
        )
        insert_width = right_edge_idx - left_edge_idx
        self.insert_centre = insert_centre
        self.left_column_idx = left_edge_idx + pixelate(0.25 * insert_width)
        self.right_column_idx = left_edge_idx + pixelate(0.75 * insert_width)

    def measure_length(self, binary_columns: np.ndarray) -> float:
        pixel_lengths = []
        for col in binary_columns.T:
            length = np.where(col)[0][0]
            pixel_lengths.append(length)

        return np.array(pixel_lengths).mean()

    def run(self) -> None:
        self.preprocess()
        _, _, _, h = self.bounding_rectangle
        roi_left = self.binary_image[
            self.insert_centre[0] : self.insert_centre[0] + pixelate(h / 5),
            self.left_column_idx - 2 : self.left_column_idx + 3,
        ]
        roi_right = self.binary_image[
            self.insert_centre[0] : self.insert_centre[0] + pixelate(h / 5),
            self.right_column_idx - 2 : self.right_column_idx + 3,
        ]
        self.length_left = self.measure_length(roi_left)
        self.length_right = self.measure_length(roi_right)
        self.pixel_difference = np.abs(self.length_left - self.length_right)
