from pathlib import Path

import numpy as np
from pydicom.uid import generate_uid

from orthovision.data.catalog import build_studies
from orthovision.data.config import DataConfig
from orthovision.data.discovery import discover_dicom_headers
from orthovision.data.models import SliceRecord, empty_spatial
from orthovision.data.spatial import detect_plane, order_slices, slice_normal
from tests.synth_dicom import IOP_AXIAL, IOP_CORONAL, IOP_SAGITTAL, write_mr_slice


def _slice(iop, ipp, inst: int, sop: str) -> SliceRecord:
    spatial = type(empty_spatial())(
        rows=8,
        columns=8,
        pixel_spacing=(0.5, 0.5),
        slice_thickness=3.0,
        spacing_between_slices=3.0,
        image_orientation_patient=iop,
        image_position_patient=ipp,
    )
    return SliceRecord(
        sop_instance_uid=sop,
        source_path=Path(f"{sop}.dcm"),
        instance_number=inst,
        transfer_syntax_uid="1.2.840.10008.1.2.1",
        spatial=spatial,
        slice_position=None,
        number_of_frames=None,
    )


def test_spatial_sort_ignores_scrambled_instance_numbers():
    iop = IOP_SAGITTAL
    n = slice_normal(iop)
    assert n is not None
    # Positions along +X for sagittal (normal ~ -X or +X)
    slices = [
        _slice(iop, (30.0, 0.0, 0.0), inst=1, sop="c"),
        _slice(iop, (10.0, 0.0, 0.0), inst=2, sop="a"),
        _slice(iop, (20.0, 0.0, 0.0), inst=3, sop="b"),
    ]
    ordered, audit = order_slices(slices)
    assert audit.method == "spatial"
    xs = [s.spatial.image_position_patient[0] for s in ordered]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)
    # monotonic along projected coordinate
    pos = [s.slice_position for s in ordered]
    assert pos == sorted(pos)


def test_fallback_instance_number_when_no_ipp():
    slices = [
        _slice(None, None, inst=3, sop="z"),
        _slice(None, None, inst=1, sop="a"),
        _slice(None, None, inst=2, sop="m"),
    ]
    ordered, audit = order_slices(slices)
    assert audit.method == "instance_number"
    assert [s.instance_number for s in ordered] == [1, 2, 3]
    assert any("FALLBACK" in n for n in audit.notes)


def test_fallback_sop_when_no_instance_number():
    slices = [
        _slice(None, None, inst=None, sop="b"),
        _slice(None, None, inst=None, sop="a"),
    ]
    ordered, audit = order_slices(slices)
    assert audit.method == "sop_instance_uid"
    assert [s.sop_instance_uid for s in ordered] == ["a", "b"]


def test_plane_from_iop_not_description():
    sag = [_slice(IOP_SAGITTAL, (0, 0, 0), 1, "s")]
    cor = [_slice(IOP_CORONAL, (0, 0, 0), 1, "c")]
    ax = [_slice(IOP_AXIAL, (0, 0, 0), 1, "a")]
    assert detect_plane(sag, series_description="CORONAL PD").plane == "sagittal"
    assert detect_plane(cor, series_description="SAG").plane == "coronal"
    assert detect_plane(ax, series_description=None).plane == "axial"


def test_oblique_is_unknown():
    # 45-degree-ish row/col producing an oblique normal
    iop = (0.7071, 0.7071, 0.0, 0.0, 0.0, -1.0)
    audit = detect_plane([_slice(iop, (0, 0, 0), 1, "o")])
    assert audit.plane == "unknown"


def test_missing_orientation_unknown_plane():
    audit = detect_plane([_slice(None, (0, 0, 0), 1, "x")])
    assert audit.plane == "unknown"
    assert audit.method == "unknown"


def test_end_to_end_order_on_written_files(tmp_path: Path, data_cfg: DataConfig):
    raw = tmp_path / "raw"
    study, series = generate_uid(), generate_uid()
    # Write in reverse spatial order with matching reverse instance numbers
    for inst, x in [(1, 20.0), (2, 10.0), (3, 0.0)]:
        write_mr_slice(
            raw / f"{inst}.dcm",
            study_uid=study,
            series_uid=series,
            instance_number=inst,
            ipp=(x, 0.0, 0.0),
            iop=IOP_SAGITTAL,
        )
    headers, issues, _ = discover_dicom_headers(raw, num_workers=1)
    studies = build_studies(headers, issues, data_cfg)
    ser = studies[0].series[0]
    assert ser.ordering_audit.method == "spatial"
    assert ser.plane == "sagittal"
    xs = [sl.spatial.image_position_patient[0] for sl in ser.slices]
    assert np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0)
