from scipy.signal import find_peaks, peak_prominences
import numpy as np
from typing import Protocol
from enum import Enum


# Row, column distances from the centre of the ACR phantom in milimetres.
LANDMARKS_MM = np.array(
    [
        [32.08333445, -21.1979174],
        [38.385418, -13.75000048],
        [32.65625113, 2.29166675],
        [38.385418, 9.16666698],
        [33.22916782, 25.78125089],
        [38.385418, 32.08333445],
    ]
)

HOLE_ARRAY_SPACINGS = (1.1, 1.0, 0.9)  # in mm
NUM_ARRAY_ROWS = 4
NUM_ARRAY_COLS = 4
PEAK_HEIGHT_FRACTION = 0.5


class HoleArray(Enum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2


class PreprocessedSlice(Protocol):
    ...


class HCSRTest:
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.landmarks = LANDMARKS_MM / self.pixel_spacing[0]
        self.hole_spacings = np.array(HOLE_ARRAY_SPACINGS) / self.pixel_spacing[0]
        self._origin = None
        self.upper_left_landmarks = dict()
        self.lower_right_landmarks = dict()
        self.metric_values = dict()

    def adjust_centre(self) -> tuple[int, int]:
        """
        Adjust the centre coordinates of phantom if the bounding box is not
        drawn correctly due to air bubble.
        """
        x, y, w, h = self.bounding_rectangle
        initial_centre = np.array(
            (np.round(y + (h - 1) / 2), np.round(x + (w - 1) / 2))
        ).astype(int)
        # Use the boundaries of the ramp insert to adjust centre.
        top_bound_search = self.binary_image[: initial_centre[0], initial_centre[1]]
        bottom_bound_search = self.binary_image[initial_centre[0] :, initial_centre[1]]
        top_bound = np.where(top_bound_search)[0][-1]
        bottom_bound = np.where(bottom_bound_search)[0][0]
        ramp_insert_height = initial_centre[0] + bottom_bound - top_bound
        centre_row = top_bound + int(np.round(ramp_insert_height / 2))
        # Based on empirical results, adjustment of 1mm is needed
        pixel_adjustment = int(np.round(1 / self.pixel_spacing[1]))
        centre_row -= pixel_adjustment
        centre_col = initial_centre[1]
        return (centre_row, centre_col)

    @property
    def origin(self) -> tuple[int, int]:
        if self._origin is None:
            x, y, w, h = self.bounding_rectangle
            if abs(w - h) / np.mean((w, h)) < 0.01:
                self._origin = np.array(
                    (np.round(y + (h - 1) / 2), np.round(x + (w - 1) / 2))
                ).astype(int)
            else:
                self._origin = self.adjust_centre()
        return self._origin

    def test_array_pair(self, hole_array: HoleArray) -> None:
        horizontal_profiles = self.get_profiles_ul(hole_array)
        test_horizontal = self.test_profiles(horizontal_profiles)
        vertical_profiles = self.get_profiles_lr(hole_array)
        test_vertical = self.test_profiles(vertical_profiles)
        results_dict = {"horizontal": test_horizontal, "vertical": test_vertical}
        self.metric_values[hole_array.name] = results_dict

    def test_profiles(self, profiles: np.array) -> list[np.array]:
        num_profiles = len(profiles)
        profile_metric_vals = []
        for i in range(num_profiles):
            profile = profiles[i]
            required_height = PEAK_HEIGHT_FRACTION * profile.max()
            peaks, _ = find_peaks(profile, height=required_height)
            prominence_values = peak_prominences(profile, peaks)[0]
            # Metric value is the height of the local minima divided by its neighbouring peak
            metric_values = (profile[peaks] - prominence_values) / profile[peaks]
            profile_metric_vals.append(metric_values)
        return metric_values

    def get_profiles_ul(self, hole_array: HoleArray) -> np.array:
        """
        Get the 4 horizontal line profiles for the hole arrangement in the
        upper left section of one of the hole array sections.
        """
        # Length of line profile in pixels
        profile_length = int(
            2
            * HOLE_ARRAY_SPACINGS[hole_array.value]
            / self.pixel_spacing[1]
            * NUM_ARRAY_ROWS
        )
        landmark = self.landmarks[2 * hole_array.value, :]
        # line_profile_rows_start is relative to image centre
        line_profile_rows_start = np.tile(landmark, (4, 1))
        line_profile_rows_start[:, 0] += (
            2 * self.hole_spacings[hole_array.value] * np.arange(NUM_ARRAY_ROWS)
        )
        line_profile_rows_start = np.round(line_profile_rows_start).astype(int)
        # Get line profile starts relative to top left corner of image.
        line_profile_rows_start[:, 0] += self.origin[0]
        line_profile_rows_start[:, 1] += self.origin[1]
        self.upper_left_landmarks[hole_array.name] = line_profile_rows_start

        profiles_ul = []

        for i in range(NUM_ARRAY_ROWS):
            profile_row_ind = line_profile_rows_start[i, 0]
            profile_col_ind = line_profile_rows_start[i, 1]

            line_profiles = self.pixel_array[
                profile_row_ind - 1 : profile_row_ind + 2,
                profile_col_ind - 2 : profile_col_ind + profile_length,
            ]  # 2 pixel buffer for the column
            line_profile_maxima = np.max(line_profiles, axis=1)
            profile_max_ind = np.where(
                line_profile_maxima == line_profile_maxima.max()
            )[0][0]

            line_profile = line_profiles[profile_max_ind, :]
            profiles_ul.append(line_profile)

        return np.array(profiles_ul)

    def get_profiles_lr(self, hole_array: HoleArray) -> np.array:
        """
        Get the 4 vertical line profiles for the hole arrangement in the
        lower right section of one of the hole array sections.
        """
        profile_length = int(
            2
            * HOLE_ARRAY_SPACINGS[hole_array.value]
            / self.pixel_spacing[0]
            * NUM_ARRAY_ROWS
        )
        landmark = self.landmarks[2 * hole_array.value + 1, :]
        line_profile_cols_start = np.tile(landmark, (4, 1))
        line_profile_cols_start[:, 1] += (
            2 * self.hole_spacings[hole_array.value] * np.arange(NUM_ARRAY_ROWS)
        )
        line_profile_cols_start = np.round(line_profile_cols_start).astype(int)
        # Get line profile starts relative to top left corner of image.
        line_profile_cols_start[:, 0] += self.origin[0]
        line_profile_cols_start[:, 1] += self.origin[1]
        self.lower_right_landmarks[hole_array.name] = line_profile_cols_start

        profiles_lr = []

        for i in range(NUM_ARRAY_COLS):
            profile_row_ind = line_profile_cols_start[i, 0]
            profile_col_ind = line_profile_cols_start[i, 1]

            line_profiles = self.pixel_array[
                profile_row_ind - 2 : profile_row_ind + profile_length,
                profile_col_ind - 1 : profile_col_ind + 2,
            ]  # 2 pixel buffer for the row
            line_profile_maxima = np.max(line_profiles, axis=0)
            profile_max_ind = np.where(
                line_profile_maxima == line_profile_maxima.max()
            )[0][0]

            line_profile = line_profiles[:, profile_max_ind]
            profiles_lr.append(line_profile)

        return np.array(profiles_lr)

    def run(self) -> None:
        for hole_array in HoleArray:
            self.test_array_pair(hole_array)
