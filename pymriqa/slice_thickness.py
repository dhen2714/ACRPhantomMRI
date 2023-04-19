from .image_processing import PreprocessedSlice, binary_image
import numpy as np

RAMP_INSERT_WIDTH = 180  # in mm


class SliceThicknessTest:
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.top_centre = None
        self.bottom_centre = None
        self.ramp_mean_top = None
        self.ramp_mean_bottom = None
        self.ramp_binary_image = None
        self.slice_thickness = None

    def get_insert_centre(self) -> tuple[tuple[int]]:
        """Find pixel index of ramp insert centre"""
        x, y, w, h = self.bounding_rectangle
        phantom_centre = np.array(
            (np.round(y + (h - 1) / 2), np.round(x + (w - 1) / 2))
        ).astype(
            int
        )  # row, column

        # Find the centre row index of the ramp insert
        top_search = self.binary_image[: phantom_centre[0], phantom_centre[1]]
        bottom_search = self.binary_image[phantom_centre[0] :, phantom_centre[1]]

        top_edge = np.where(top_search)[0][-1]
        bottom_edge = np.where(bottom_search)[0][0]

        ramp_insert_height = phantom_centre[0] + bottom_edge - top_edge  # in pixels

        ramp_insert_centre = top_edge + ramp_insert_height / 2
        centre_top = np.round(ramp_insert_centre - ramp_insert_height / 4).astype(int)
        centre_bot = np.round(ramp_insert_centre + ramp_insert_height / 4).astype(int)
        return ((centre_top, phantom_centre[1]), (centre_bot, phantom_centre[1]))

    def get_ramp_mean(
        self,
        top_ramp_centre: tuple[int],
        bottom_ramp_centre: tuple[int],
    ) -> float:
        roi_top = self.pixel_array[
            top_ramp_centre[0] - 1 : top_ramp_centre[0] + 2,
            top_ramp_centre[1] - 30 : top_ramp_centre[1] + 31,
        ]
        roi_bottom = self.pixel_array[
            bottom_ramp_centre[0] - 1 : bottom_ramp_centre[0] + 2,
            bottom_ramp_centre[1] - 30 : bottom_ramp_centre[1] + 31,
        ]
        top_mean, bottom_mean = roi_top.mean(), roi_bottom.mean()
        self.ramp_mean_top = top_mean
        self.ramp_mean_bottom = bottom_mean
        return np.mean((top_mean, bottom_mean))

    def measure_length(self, roi: np.ndarray, half_width: int) -> float:
        lengths = []
        for row in roi:
            right_search = row[half_width:]
            right_edge = np.where(right_search == 0)[0][0]
            left_search = row[:half_width]
            left_edge = np.where(left_search == 0)[0][-1]
            pixel_length = (right_edge + half_width) - left_edge
            lengths.append(pixel_length)
        lengths = np.array(lengths) * self.pixel_spacing[0]
        return lengths.mean()

    def run(self) -> None:
        top_ramp_centre, bottom_ramp_centre = self.get_insert_centre()
        self.top_centre = top_ramp_centre
        self.bottom_centre = bottom_ramp_centre
        ramp_pixel_mean = self.get_ramp_mean(top_ramp_centre, bottom_ramp_centre)
        # self.ramp_binary_image = self.pixel_array > ramp_pixel_mean / 2
        self.ramp_binary_image = binary_image(self.pixel_array, (ramp_pixel_mean / 2))
        ramp_pixel_width = RAMP_INSERT_WIDTH / self.pixel_spacing[0]
        half_width = int(np.floor(ramp_pixel_width / 2))

        roi_top = self.ramp_binary_image[
            top_ramp_centre[0] - 1 : top_ramp_centre[0] + 2,
            top_ramp_centre[1] - half_width : top_ramp_centre[1] + half_width,
        ]
        roi_bottom = self.ramp_binary_image[
            bottom_ramp_centre[0] - 1 : bottom_ramp_centre[0] + 2,
            bottom_ramp_centre[1] - half_width : bottom_ramp_centre[1] + half_width,
        ]
        length_top = self.measure_length(roi_top, half_width)
        length_bottom = self.measure_length(roi_bottom, half_width)
        self.slice_thickness = (
            0.2 * (length_top * length_bottom) / (length_top + length_bottom)
        )
