"""Build tiny synthetic MR DICOMs for tests. Not clinical data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# LPS direction cosines (row, then column).
IOP_AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
IOP_CORONAL = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
IOP_SAGITTAL = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)


def write_mr_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    ipp: tuple[float, float, float] | None,
    iop: tuple[float, ...] | None,
    rows: int = 8,
    cols: int = 8,
    sop_uid: str | None = None,
    series_description: str | None = "PD_SAG",
    slice_thickness: float | None = 3.0,
    spacing_between_slices: float | None = 3.0,
    pixel_spacing: tuple[float, float] | None = (0.5, 0.5),
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    omit_uids: bool = False,
    laterality: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    sop = sop_uid or generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop
    if not omit_uids:
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.InstanceNumber = instance_number
    if laterality is not None:
        ds.Laterality = laterality
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if series_description is not None:
        ds.SeriesDescription = series_description
    if iop is not None:
        ds.ImageOrientationPatient = list(iop)
    if ipp is not None:
        ds.ImagePositionPatient = list(ipp)
    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)
    if slice_thickness is not None:
        ds.SliceThickness = slice_thickness
    if spacing_between_slices is not None:
        ds.SpacingBetweenSlices = spacing_between_slices
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept

    arr = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    arr = arr + np.uint16(instance_number * 10)
    ds.PixelData = arr.tobytes()
    ds.save_as(path, write_like_original=False)
    return path
