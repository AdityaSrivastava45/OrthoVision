from pathlib import Path

from pydicom.uid import generate_uid

from orthovision.data.catalog import KneeMRIDataset, build_studies, summarize_study
from orthovision.data.config import DataConfig
from orthovision.data.discovery import discover_dicom_headers
from orthovision.data.manifest import write_manifests
from orthovision.data.pixels import decode_path
from tests.synth_dicom import IOP_AXIAL, IOP_SAGITTAL, write_mr_slice


def _cfg(tmp_path: Path, data_cfg: DataConfig) -> DataConfig:
    return data_cfg


def test_inconsistent_dimensions_invalid(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    study, series = generate_uid(), generate_uid()
    write_mr_slice(raw / "a.dcm", study_uid=study, series_uid=series, instance_number=1, ipp=(0, 0, 0), iop=IOP_SAGITTAL, rows=8, cols=8)
    write_mr_slice(raw / "b.dcm", study_uid=study, series_uid=series, instance_number=2, ipp=(3, 0, 0), iop=IOP_SAGITTAL, rows=16, cols=8)
    headers, issues, _ = discover_dicom_headers(raw, num_workers=1)
    ser = build_studies(headers, issues, data_cfg)[0].series[0]
    assert ser.validation.status == "INVALID"
    assert any("inconsistent dimensions" in m for m in ser.validation.messages)


def test_duplicate_positions_warning(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    study, series = generate_uid(), generate_uid()
    for i, inst in enumerate((1, 2, 3, 4)):
        x = 0.0 if i < 2 else float(i)
        write_mr_slice(raw / f"{inst}.dcm", study_uid=study, series_uid=series, instance_number=inst, ipp=(x, 0, 0), iop=IOP_SAGITTAL)
    headers, issues, _ = discover_dicom_headers(raw, num_workers=1)
    ser = build_studies(headers, issues, data_cfg)[0].series[0]
    assert ser.validation.status in {"WARNING", "INVALID"}
    assert any("duplicate" in m.lower() for m in ser.validation.messages)


def test_missing_metadata_warning(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    study, series = generate_uid(), generate_uid()
    write_mr_slice(
        raw / "a.dcm",
        study_uid=study,
        series_uid=series,
        instance_number=1,
        ipp=None,
        iop=None,
        pixel_spacing=None,
    )
    headers, issues, _ = discover_dicom_headers(raw, num_workers=1)
    ser = build_studies(headers, issues, data_cfg)[0].series[0]
    assert ser.plane == "unknown"
    assert ser.ordering_audit.method != "spatial"
    assert ser.validation.status in {"WARNING", "INVALID"}


def test_rescale_applied_separately_from_raw(tmp_path: Path):
    path = write_mr_slice(
        tmp_path / "r.dcm",
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        instance_number=1,
        ipp=(0, 0, 0),
        iop=IOP_AXIAL,
        rescale_slope=2.0,
        rescale_intercept=5.0,
    )
    payload = decode_path(path)
    assert payload.raw.dtype == payload.raw.dtype
    assert payload.rescaled[0, 0] == payload.raw[0, 0] * 2.0 + 5.0
    assert "rescale_slope_intercept" in payload.applied


def test_study_summary_and_manifest(tmp_path: Path, data_cfg: DataConfig):
    raw = data_cfg.dataset_root
    study = generate_uid()
    sag, ax = generate_uid(), generate_uid()
    for i in range(4):
        write_mr_slice(raw / "sag" / f"{i}.dcm", study_uid=study, series_uid=sag, instance_number=i + 1, ipp=(float(i), 0, 0), iop=IOP_SAGITTAL)
    for i in range(4):
        write_mr_slice(raw / "ax" / f"{i}.dcm", study_uid=study, series_uid=ax, instance_number=i + 1, ipp=(0, 0, float(i)), iop=IOP_AXIAL)
    ds = KneeMRIDataset(data_cfg)
    ds.scan()
    summary = summarize_study(ds.get_study(study))
    assert summary.n_series == 2
    assert summary.total_instances == 8
    assert set(summary.available_planes) <= {"sagittal", "axial", "unknown"}
    paths = write_manifests(ds)
    text = paths["csv"].read_text(encoding="utf-8")
    assert "study_id" in text
    assert study in text
    assert "PixelData" not in text
