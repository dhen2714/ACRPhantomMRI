import numpy as np


def bounds_vertical(
    binary_image, column_index, row_start=0, slice_length=100
) -> tuple[tuple[int], tuple[int]]:
    """
    Measures vertical extent of non-zero pixels in a single column of a binary image.
    """
    measure_slice = binary_image[row_start : (row_start + slice_length), column_index]
    filled_pixels = np.where(measure_slice)[0]
    measure_start, measure_end = filled_pixels[0], filled_pixels[-1]
    return (
        (row_start + measure_start, column_index),
        (row_start + measure_end, column_index),
    )


def bounds_horizontal(
    binary_image, row_index, column_start=0, slice_length=100
) -> tuple[tuple[int], tuple[int]]:
    """
    Measures vertical extent of non-zero pixels in a single column of a binary image.
    """
    measure_slice = binary_image[
        row_index, column_start : (column_start + slice_length)
    ]
    filled_pixels = np.where(measure_slice)[0]
    measure_start, measure_end = filled_pixels[0], filled_pixels[-1]
    return (
        (row_index, column_start + measure_start),
        (row_index, column_start + measure_end),
    )
