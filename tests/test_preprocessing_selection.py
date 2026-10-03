import numpy as np
import pytest

from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.discovery import SeriesInfo
from orthovision.preprocessing.geometry import SeriesGeometry, SliceRow
from orthovision.preprocessing.selection import rank_candidates


def fake_series(
    uid: str,
    *,
    plane="Sagittal",
    fluid=1,
    n_slices=30,
    rows=384,
    columns=384,
    completeness=1.0,
    spatial=True,
    derived="sagittal",
):
    geometry = SeriesGeometry(
        ordering_method="spatial" if spatial else "instance_number",
        normal=(1.0, 0.0, 0.0) if spatial else None,
        positions_mm=[0.0] * n_slices if spatial else [None] * n_slices,
        median_gap_mm=3.0 if spatial else None,
        gap_cv=0.001 if spatial else None,
        duplicate_positions=0,
        span_mm=90.0 if spatial else None,
        laterality="L",
        patient_position="FFS",
    )
    return SeriesInfo(
        study_uid="study",
        series_uid=uid,
        plane_csv=plane,
        plane_derived=derived if spatial else None,
        fluid_sensitive=fluid,
        fat_suppression=fluid,
        n_slices=n_slices,
        rows=rows,
        columns=columns,
        pixel_spacing_mm=(0.4, 0.4),
        slice_thickness_mm=3.0,
        slice_spacing_mm=3.0 if spatial else None,
        orientation_normal=(1.0, 0.0, 0.0) if spatial else None,
        ordering_method="spatial" if spatial else "instance_number",
        laterality="L",
        patient_position="FFS",
        series_description="pd_tse_fs_sag",
        modality="MR",
        transfer_syntax_uid="1.2.840.10008.1.2.1",
        completeness=completeness,
        slices_ordered=[],
        geometry=geometry,
        source_paths=[],
    )


CFG = PreprocessConfig()


def test_prefers_matching_fluid_preference():
    a = fake_series("aaa", fluid=0)
    b = fake_series("bbb", fluid=1)
    report = rank_candidates([a, b], plane="Sagittal", fluid_preference=1, cfg=CFG)
    assert report.selected_series_uid == "bbb"
    assert report.candidates[0].score > report.candidates[1].score


def test_hard_plane_filter_excludes_wrong_plane():
    ax = fake_series("axial", plane="Axial", derived="axial")
    report = rank_candidates([ax], plane="Sagittal", fluid_preference=None, cfg=CFG)
    assert report.selected_series_uid is None
    assert report.candidates == []


def test_more_slices_and_consistency_win():
    small = fake_series("small", n_slices=8)
    big = fake_series("big", n_slices=40)
    broken = fake_series("broken", n_slices=40, spatial=False)
    report = rank_candidates([small, broken, big], plane="Sagittal", fluid_preference=None, cfg=CFG)
    assert report.selected_series_uid == "big"
    scores = {c.series_uid: c.score for c in report.candidates}
    assert scores["big"] > scores["broken"] > scores["small"] or scores["big"] > scores["broken"]


def test_tie_break_is_deterministic_by_uid():
    a = fake_series("zzz", n_slices=30)
    b = fake_series("aaa", n_slices=30)
    r1 = rank_candidates([a, b], plane="Sagittal", fluid_preference=None, cfg=CFG)
    r2 = rank_candidates([b, a], plane="Sagittal", fluid_preference=None, cfg=CFG)
    assert r1.selected_series_uid == r2.selected_series_uid == "aaa"


def test_reasons_present_for_every_candidate():
    report = rank_candidates(
        [fake_series("c1"), fake_series("c2", fluid=0)],
        plane="Sagittal",
        fluid_preference=1,
        cfg=CFG,
    )
    for c in report.candidates:
        assert c.reasons
        assert c.components


def test_weights_change_winner():
    few_but_complete = fake_series("few", n_slices=10, completeness=1.0)
    many_incomplete = fake_series("many", n_slices=60, completeness=0.5)
    base = PreprocessConfig(ranking_weights={"fluid": 0.0, "slices": 5.0, "resolution": 0.0, "completeness": 0.0, "consistency": 0.0})
    flipped = PreprocessConfig(ranking_weights={"fluid": 0.0, "slices": 0.0, "resolution": 0.0, "completeness": 5.0, "consistency": 0.0})
    assert rank_candidates([few_but_complete, many_incomplete], plane="Sagittal", fluid_preference=None, cfg=base).selected_series_uid == "many"
    assert rank_candidates([few_but_complete, many_incomplete], plane="Sagittal", fluid_preference=None, cfg=flipped).selected_series_uid == "few"


def test_plane_derived_mismatch_penalised():
    mismatch = fake_series("mismatch", derived="coronal")
    match = fake_series("match")
    report = rank_candidates([mismatch, match], plane="Sagittal", fluid_preference=None, cfg=CFG)
    comp = {c.series_uid: c.components for c in report.candidates}
    assert comp["match"]["consistency"] > comp["mismatch"]["consistency"]
