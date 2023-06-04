from .image_processing import PreprocessedSlice
from .image_processing import binary_image, bounding_rectangle
from .measurement import bounds_horizontal, bounds_vertical
import cv2


class GeometricAccuracyTest:
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        self.pixel_array = preprocessed_slice.pixel_array
        self.pixel_spacing = preprocessed_slice.pixel_spacing
        self.binary_image = preprocessed_slice.binary_image
        self.binary_threshold = preprocessed_slice.binary_threshold
        self.bounding_rectangle = preprocessed_slice.bounding_rectangle
        self.water_mean = preprocessed_slice.water_mean
        self.measurement_bounds = []
        self.measurements = []


class LocaliserTest(GeometricAccuracyTest):
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        super().__init__(preprocessed_slice)
        self.measurement_locations = [0.4, 0.9]

    def run(self) -> None:
        x, y, w, h = self.bounding_rectangle
        measurement_columns = [
            int(x + fraction * w) for fraction in self.measurement_locations
        ]
        pixel_spacing = self.pixel_spacing
        for measure_col in measurement_columns:
            bounds = bounds_vertical(
                self.binary_image, measure_col, row_start=y, slice_length=h
            )
            pixel_height = bounds[1][0] - bounds[0][0]
            self.measurements.append(pixel_spacing[0] * pixel_height)
            self.measurement_bounds.append(bounds)


class AxialTest(GeometricAccuracyTest):
    def __init__(self, preprocessed_slice: PreprocessedSlice) -> None:
        super().__init__(preprocessed_slice)
        self.rotated_image = None
        self.binary_image_diagonal = None
        self.bounding_rectangle_diagonal = None
        self.measurement_bounds_diagonal = None
        self.measurement_keys = [
            "vertical",
            "horizontal",
            "vertical_diagonal",
            "horizontal_diagonal",
        ]

    def preprocess_diagonal(self) -> None:
        image_height, image_width = self.pixel_array.shape
        x, y, w, h = self.bounding_rectangle
        iso_x, iso_y = x + w // 2, y + h // 2
        R = cv2.getRotationMatrix2D((iso_x, iso_y), angle=45, scale=1)
        self.rotated_image = cv2.warpAffine(
            self.pixel_array, R, (image_width, image_height)
        )
        self.binary_image_diagonal = binary_image(
            self.rotated_image, self.binary_threshold
        )
        self.bounding_rectangle_diagonal = bounding_rectangle(
            self.binary_image_diagonal
        )

    def run(self) -> None:
        self.preprocess_diagonal()
        pixel_spacing = self.pixel_spacing
        for bounding_rect, binary_image in zip(
            (self.bounding_rectangle, self.bounding_rectangle_diagonal),
            (self.binary_image, self.binary_image_diagonal),
        ):
            x, y, w, h = bounding_rect
            measure_col = int(x + 0.5 * w)
            measure_row = int(y + 0.5 * h)
            vbounds = bounds_vertical(
                binary_image, measure_col, row_start=y, slice_length=h
            )
            hbounds = bounds_horizontal(
                binary_image, measure_row, column_start=x, slice_length=w
            )
            pixel_height = vbounds[1][0] - vbounds[0][0]
            pixel_width = hbounds[1][1] - hbounds[0][1]
            self.measurements.append(pixel_spacing[0] * pixel_height)
            self.measurements.append(pixel_spacing[1] * pixel_width)
            self.measurement_bounds.append(vbounds)
            self.measurement_bounds.append(hbounds)
