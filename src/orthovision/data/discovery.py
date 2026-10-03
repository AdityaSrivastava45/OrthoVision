"""Recursive DICOM discovery that does not assume extensions or perfect files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pydicom.dataset import Dataset, FileDataset
from pydicom.errors import InvalidDicomError
from pydicom.misc import is_dicom

from orthovision.data.logutil import get_logger
from orthovision.data.models import FileIssue, ScanStage, SpatialMetadata, empty_spatial
from orthovision.data.tags import (
    get_float,
    get_floats,
    get_int,
    get_str,
    get_uid,
    transfer_syntax_uid,
)

log = get_logger("discovery")

_SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "venv", ".ipynb_checkpoints"}
_MIN_DICOM_BYTES = 132


@dataclass
class HeaderRecord:
    path: Path
    dataset: Dataset
    study_uid: str | None
    series_uid: str | None
    sop_uid: str | None
    instance_number: int | None
    transfer_syntax_uid: str | None
    spatial: SpatialMetadata
    number_of_frames: int | None
    metadata: dict[str, str | int | float | tuple | None]


def _spatial_from(ds: Dataset) -> SpatialMetadata:
    ipp = get_floats(ds, "ImagePositionPatient", expected=3)
    iop = get_floats(ds, "ImageOrientationPatient")
    spacing = get_floats(ds, "PixelSpacing")
    pixel_spacing = None
    if spacing is not None and len(spacing) >= 2:
        pixel_spacing = (float(spacing[0]), float(spacing[1]))
    return SpatialMetadata(
        rows=get_int(ds, "Rows"),
        columns=get_int(ds, "Columns"),
        pixel_spacing=pixel_spacing,
        slice_thickness=get_float(ds, "SliceThickness"),
        spacing_between_slices=get_float(ds, "SpacingBetweenSlices"),
        image_orientation_patient=iop if iop and len(iop) >= 6 else None,
        image_position_patient=ipp,
    )


def extract_header_fields(ds: Dataset) -> dict[str, str | int | float | tuple | None]:
    keys = (
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "SOPClassUID",
        "SeriesDescription",
        "SeriesNumber",
        "Modality",
        "BodyPartExamined",
        "ProtocolName",
        "SequenceName",
        "SliceThickness",
        "SpacingBetweenSlices",
        "PhotometricInterpretation",
        "PatientID",
        "StudyDate",
        "StudyDescription",
        "Manufacturer",
        "ManufacturerModelName",
        "Laterality",
        "PatientPosition",
    )
    out: dict[str, str | int | float | tuple | None] = {}
    for key in keys:
        if key == "SeriesNumber":
            out[key] = get_int(ds, key)
        elif key in ("SliceThickness", "SpacingBetweenSlices"):
            out[key] = get_float(ds, key)
        else:
            out[key] = get_str(ds, key)
    out["TransferSyntaxUID"] = transfer_syntax_uid(ds)
    out["ImageOrientationPatient"] = get_floats(ds, "ImageOrientationPatient")
    out["ImagePositionPatient"] = get_floats(ds, "ImagePositionPatient", expected=3)
    out["PixelSpacing"] = get_floats(ds, "PixelSpacing")
    out["Rows"] = get_int(ds, "Rows")
    out["Columns"] = get_int(ds, "Columns")
    out["InstanceNumber"] = get_int(ds, "InstanceNumber")
    out["NumberOfFrames"] = get_int(ds, "NumberOfFrames")
    out["SamplesPerPixel"] = get_int(ds, "SamplesPerPixel")
    out["BitsAllocated"] = get_int(ds, "BitsAllocated")
    out["BitsStored"] = get_int(ds, "BitsStored")
    out["HighBit"] = get_int(ds, "HighBit")
    out["PixelRepresentation"] = get_int(ds, "PixelRepresentation")
    out["RescaleSlope"] = get_float(ds, "RescaleSlope")
    out["RescaleIntercept"] = get_float(ds, "RescaleIntercept")
    return out


def _read_header(path: Path) -> tuple[HeaderRecord | None, FileIssue | None, str]:
    """Return (header, issue, kind) where kind is dicom|not_dicom|unreadable."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, FileIssue(str(path), ScanStage.DISCOVERY.value, f"stat failed: {exc}", type(exc).__name__, "unreadable"), "unreadable"

    preamble = False
    try:
        preamble = is_dicom(path)
    except OSError as exc:
        return None, FileIssue(str(path), ScanStage.DISCOVERY.value, f"is_dicom failed: {exc}", type(exc).__name__, "unreadable"), "unreadable"

    if size < _MIN_DICOM_BYTES and not preamble:
        return None, None, "not_dicom"

    ds: FileDataset | Dataset | None = None
    try:
        from pydicom import dcmread

        ds = dcmread(path, stop_before_pixels=True, force=False)
    except InvalidDicomError:
        ds = None
    except Exception as exc:  # noqa: BLE001 — one bad file must not kill the scan
        if preamble:
            return (
                None,
                FileIssue(str(path), ScanStage.DISCOVERY.value, f"DICOM header read failed: {exc}", type(exc).__name__, "unreadable"),
                "unreadable",
            )
        return None, None, "not_dicom"

    if ds is None and not preamble:
        # Files without a DICM preamble: accept only if forced read yields SOP UIDs.
        try:
            from pydicom import dcmread

            forced = dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            return None, None, "not_dicom"
        if get_uid(forced, "SOPInstanceUID") is None and get_uid(forced, "SOPClassUID") is None:
            return None, None, "not_dicom"
        ds = forced
        log.debug("Accepted preamble-less DICOM: %s", path)
    elif ds is None and preamble:
        return (
            None,
            FileIssue(str(path), ScanStage.DISCOVERY.value, "File has DICM preamble but header could not be parsed", "InvalidDicomError", "unreadable"),
            "unreadable",
        )

    assert ds is not None
    meta = extract_header_fields(ds)
    rec = HeaderRecord(
        path=path,
        dataset=ds,
        study_uid=get_uid(ds, "StudyInstanceUID"),
        series_uid=get_uid(ds, "SeriesInstanceUID"),
        sop_uid=get_uid(ds, "SOPInstanceUID") or f"MISSING_SOP::{path}",
        instance_number=get_int(ds, "InstanceNumber"),
        transfer_syntax_uid=transfer_syntax_uid(ds),
        spatial=_spatial_from(ds),
        number_of_frames=get_int(ds, "NumberOfFrames"),
        metadata=meta,
    )
    return rec, None, "dicom"


def iter_candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")
    files: list[Path] = []
    for dirpath, dirnames, filenames in root.walk() if hasattr(root, "walk") else _walk_fallback(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
        for name in filenames:
            files.append(dirpath / name)
    files.sort()
    return files


def _walk_fallback(root: Path):
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


def discover_dicom_headers(
    root: Path,
    *,
    num_workers: int = 4,
) -> tuple[list[HeaderRecord], list[FileIssue], dict[str, int]]:
    """Scan *root* recursively. Never raises for a single bad file."""
    paths = iter_candidate_files(root)
    headers: list[HeaderRecord] = []
    issues: list[FileIssue] = []
    counts = {"files_seen": len(paths), "dicom_valid": 0, "dicom_unreadable": 0, "not_dicom": 0}

    workers = max(1, int(num_workers))
    log.info("Scanning %s files under %s (workers=%s)", len(paths), root, workers)

    def job(p: Path) -> tuple[HeaderRecord | None, FileIssue | None, str]:
        return _read_header(p)

    if workers == 1:
        results = [job(p) for p in paths]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(job, p): p for p in paths}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    issues.append(FileIssue(str(p), ScanStage.DISCOVERY.value, f"worker crashed: {exc}", type(exc).__name__, "unreadable"))
                    counts["dicom_unreadable"] += 1

    if workers != 1:
        # as_completed is unordered; re-sort by path for deterministic grouping later
        pass

    ordered: list[tuple[HeaderRecord | None, FileIssue | None, str]]
    if workers == 1:
        ordered = results
    else:
        # results already collected; we lost path order — sort headers later
        ordered = results

    for rec, issue, kind in ordered:
        if kind == "dicom" and rec is not None:
            headers.append(rec)
            counts["dicom_valid"] += 1
            if rec.study_uid is None or rec.series_uid is None:
                issues.append(
                    FileIssue(
                        str(rec.path),
                        ScanStage.GROUPING.value,
                        "DICOM file missing StudyInstanceUID and/or SeriesInstanceUID",
                        None,
                        "dicom",
                    )
                )
        elif kind == "unreadable":
            counts["dicom_unreadable"] += 1
            if issue:
                issues.append(issue)
        else:
            counts["not_dicom"] += 1

    headers.sort(key=lambda h: str(h.path))
    log.info(
        "Discovery complete files_seen=%s dicom_valid=%s unreadable=%s not_dicom=%s",
        counts["files_seen"],
        counts["dicom_valid"],
        counts["dicom_unreadable"],
        counts["not_dicom"],
    )
    return headers, issues, counts
