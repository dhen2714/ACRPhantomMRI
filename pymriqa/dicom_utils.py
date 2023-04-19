import pydicom
from pydicom.fileset import FileSet
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ACRT2Container:
    """Container class for sorting slices in T2 directories."""

    echo_time: int = None
    slices: list = None

    def load_slices(self, slice_list: list[str | Path]) -> None:
        """
        slice_list is a list of dicom file paths for slices that share an
        echo time.
        """
        slice_nums = {}
        for filepath in slice_list:
            dcm = pydicom.dcmread(filepath)
            slice_nums[filepath] = int(dcm.InstanceNumber)

        argsorted = sorted(slice_list, key=lambda k: slice_nums[k])
        self.slices = argsorted


def get_fileset(directory_path: str | Path) -> FileSet:
    """
    Returns a pydicom FileSet for slices in a directory. Useful when grouping
    slices that share common attributes.
    For example, FileSet.find(EchoTime=80, load=True) will return a list of
    pydicom.FileInstance objects for dicom slices where the echo time is 80 ms.

    To read a FileInstance object, call pydicom.dcmread(FileInstance.path).
    """
    fs = FileSet()
    for fname in Path(directory_path).iterdir():
        dcm = pydicom.dcmread(fname)
        fs.add(dcm)

    return fs
