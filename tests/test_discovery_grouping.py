from pathlib import Path

from pydicom.uid import generate_uid

from orthovision.data.catalog import KneeMRIDataset, build_studies
from orthovision.data.config import DataConfig
from orthovision.data.discovery import discover_dicom_headers
from tests.synth_dicom import IOP_SAGITTAL, write_mr_slice


def test_discovers_dicom_without_extension_and_skips_junk(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    study, series = generate_uid(), generate_uid()
    write_mr_slice(
        raw / "nested" / "noext",
        study_uid=study,
        series_uid=series,
        instance_number=1,
        ipp=(0, 0, 0),
        iop=IOP_SAGITTAL,
    )
    (raw / "readme.txt").write_text("not a dicom", encoding="utf-8")
    (raw / "tiny.bin").write_bytes(b"xxx")
    (raw / "nested" / "corrupt.dcm").write_bytes(b"DICM" + b"\x00" * 200)

    headers, issues, counts = discover_dicom_headers(raw, num_workers=1)
    assert counts["files_seen"] >= 4
    assert counts["dicom_valid"] == 1
    assert counts["not_dicom"] >= 2
    assert headers[0].study_uid == study
    assert headers[0].series_uid == series


def test_malformed_preamble_is_recorded_not_crash(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Valid DICM magic at offset 128 but garbage thereafter
    payload = b"\x00" * 128 + b"DICM" + b"\xff" * 80
    (raw / "bad.dcm").write_bytes(payload)
    headers, issues, counts = discover_dicom_headers(raw, num_workers=1)
    assert issues, "malformed file must be recorded as an issue, never silently skipped"
    assert all(i.stage in ("discovery", "grouping") for i in issues)
    usable = [h for h in headers if h.study_uid or h.series_uid]
    assert not usable


def test_grouping_by_study_and_series_uids(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    s1, s2 = generate_uid(), generate_uid()
    a, b = generate_uid(), generate_uid()
    for i in range(3):
        write_mr_slice(raw / f"a{i}.dcm", study_uid=s1, series_uid=a, instance_number=i + 1, ipp=(float(i), 0, 0), iop=IOP_SAGITTAL)
    for i in range(2):
        write_mr_slice(raw / f"b{i}.dcm", study_uid=s1, series_uid=b, instance_number=i + 1, ipp=(0, float(i), 0), iop=IOP_SAGITTAL)
    write_mr_slice(raw / "other.dcm", study_uid=s2, series_uid=generate_uid(), instance_number=1, ipp=(0, 0, 0), iop=IOP_SAGITTAL)

    headers, issues, _ = discover_dicom_headers(raw, num_workers=1)
    studies = build_studies(headers, issues, data_cfg)
    by_id = {st.study_id: st for st in studies}
    assert set(by_id) == {s1, s2}
    assert {ser.series_id for ser in by_id[s1].series} == {a, b}
    assert by_id[s1].get_series(a).slice_count == 3
    assert by_id[s2].get_series(by_id[s2].series[0].series_id).slice_count == 1
