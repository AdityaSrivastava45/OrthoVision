"""End-to-end Stage-2 tests on synthetic DICOM studies (no real data needed)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from orthovision.data.discovery import discover_dicom_headers
from orthovision.preprocessing.cache import series_cache_key
from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.dataset import KneeStudyDataset
from orthovision.preprocessing.discovery import build_series_info, discover_studies
from orthovision.preprocessing.geometry import SliceRow, order_slices
from orthovision.preprocessing.pipeline import StudyPreprocessor
from orthovision.preprocessing.qc import SeriesQCError, validate_series
from tests.synth_dicom import IOP_AXIAL, IOP_CORONAL, IOP_SAGITTAL, write_mr_slice


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]


def write_const_slice(path, *, study_uid, series_uid, instance_number, ipp, iop,
                      rows=32, cols=32, value=100, laterality=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mr_slice(
        path,
        study_uid=study_uid,
        series_uid=series_uid,
        instance_number=instance_number,
        ipp=ipp,
        iop=iop,
        rows=rows,
        cols=cols,
        laterality=laterality,
    )
    from pydicom import dcmread
    ds = dcmread(path)
    arr = np.full((rows, cols), value, dtype=np.uint16)
    arr[0, 0] = 0
    ds.PixelData = arr.tobytes()
    if hasattr(ds, "file_meta"):
        pass
    ds.save_as(path, enforce_file_format=True)
    return path


def build_synthetic_study(raw: Path, study_uid: str) -> dict[str, str]:
    """One sagittal FS (reversed InstanceNumbers), one coronal, one axial."""
    uids = {}
    sag = "1.2.826.999.1"
    for k in range(7):
        pos = -9.0 + 3.0 * k
        inst = 7 - k
        write_const_slice(
            raw / "train_series" / sag / f"{study_uid}.{k}.dcm",
            study_uid=study_uid,
            series_uid=sag,
            instance_number=inst,
            ipp=(pos, 0.0, 0.0),
            iop=IOP_SAGITTAL,
            laterality="L",
            value=10 * (6 - k) + 5,
        )
    uids["Sagittal"] = sag

    cor = "1.2.826.999.2"
    for k in range(6):
        write_const_slice(
            raw / "train_series" / cor / f"{study_uid}.c{k}.dcm",
            study_uid=study_uid,
            series_uid=cor,
            instance_number=k + 1,
            ipp=(0.0, -7.0 + 2.0 * k, 0.0),
            iop=IOP_CORONAL,
            rows=48,
            cols=48,
            value=20 * k + 3,
        )
    uids["Coronal"] = cor

    ax = "1.2.826.999.3"
    for k in range(8):
        write_const_slice(
            raw / "train_series" / ax / f"{study_uid}.a{k}.dcm",
            study_uid=study_uid,
            series_uid=ax,
            instance_number=k + 1,
            ipp=(0.0, 0.0, -10.0 + 2.5 * k),
            iop=IOP_AXIAL,
            rows=24,
            cols=24,
            value=5 * k + 1,
        )
    uids["Axial"] = ax
    return uids


def write_series_csv(tmp: Path, study_uid: str, uids: dict[str, str]) -> Path:
    rows = []
    fluid_map = {"Sagittal": 1, "Coronal": 1, "Axial": 0}
    plane_map = {"Sagittal": "Sagittal", "Coronal": "Coronal", "Axial": "Axial"}
    for plane, uid in uids.items():
        rows.append({
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": uid,
            "Fluid_Sensitive": fluid_map[plane],
            "Fat_Suppression": fluid_map[plane],
            "Anatomical_Plane": plane_map[plane],
        })
    p = tmp / "train_series.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def make_header_cache(tmp: Path, raw: Path) -> Path:
    headers, issues, counts = discover_dicom_headers(raw, num_workers=1)
    assert counts["dicom_valid"] > 0
    fields = [
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription",
        "Modality", "PhotometricInterpretation", "TransferSyntaxUID", "Rows", "Columns",
        "InstanceNumber", "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing",
        "SliceThickness", "SpacingBetweenSlices", "RescaleSlope", "RescaleIntercept",
        "Laterality", "PatientPosition",
    ]
    rows = []
    for h in headers:
        row = {"path": h.path.relative_to(raw).as_posix()}
        for f in fields:
            v = h.metadata.get(f)
            if isinstance(v, tuple):
                v = ";".join(f"{x:.8g}" for x in v)
            row[f] = v
        rows.append(row)
    cache = tmp / "headers.csv"
    pd.DataFrame(rows).to_csv(cache, index=False)
    return cache


@pytest.fixture
def synthetic_env(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    study_uid = "1.2.826.555.1"
    uids = build_synthetic_study(raw, study_uid)
    series_csv = write_series_csv(tmp_path, study_uid, uids)
    headers_csv = make_header_cache(tmp_path, raw)

    labels_rows = [{**{c: np.nan for c in LABELS}, "StudyInstanceUID": study_uid, "Report": ""}]
    labels_rows[0]["ACL"] = 1.0
    labels_rows[0]["Effusion"] = 0.0
    labels_csv = tmp_path / "train.csv"
    pd.DataFrame(labels_rows).to_csv(labels_csv, index=False)

    return {
        "raw": raw,
        "tmp": tmp_path,
        "study_uid": study_uid,
        "uids": uids,
        "series_csv": series_csv,
        "headers_csv": headers_csv,
        "labels_csv": labels_csv,
    }


def test_discovery_groups_by_uid_and_orders_physically(synthetic_env):
    env = synthetic_env
    studies = discover_studies(env["headers_csv"], env["series_csv"])
    assert list(studies) == [env["study_uid"]]
    infos = studies[env["study_uid"]]
    assert len(infos) == 3
    sag = next(s for s in infos if s.series_uid == env["uids"]["Sagittal"])
    assert sag.ordering_method == "spatial"
    positions = sag.geometry.positions_mm
    assert positions == sorted(positions)
    assert sag.laterality == "L"


def test_pipeline_produces_triplets_with_provenance(synthetic_env, tmp_path):
    env = synthetic_env
    cfg = PreprocessConfig(image_size=(64, 64), num_samples_per_plane=4, cache_enabled=False)
    studies = discover_studies(env["headers_csv"], env["series_csv"], cfg)
    prep = StudyPreprocessor(cfg, data_root=env["raw"], cache_dir=tmp_path / "cache")
    result = prep.process_study(env["study_uid"], studies[env["study_uid"]])

    assert len(result.samples) == 12
    by_plane = {}
    for sample in result.samples:
        by_plane.setdefault(sample.metadata["plane_csv"], []).append(sample)
    assert set(by_plane) == {"Sagittal", "Coronal", "Axial"}

    sag_samples = sorted(by_plane["Sagittal"], key=lambda s: s.metadata["center_slice_index"])
    first = sag_samples[0]
    meta = first.metadata
    for key in [
        "study_uid", "series_uid", "plane_derived", "laterality", "ordering_method",
        "slice_normal_lps", "center_slice_index", "slice_indices",
        "physical_positions_mm", "channel_roles", "sop_instance_uids",
        "source_paths", "normalization_stats", "resize_meta", "config_fingerprint",
    ]:
        assert key in meta, key
    assert meta["channel_roles"] == ["previous", "center", "next"]
    pp = meta["physical_positions_mm"]
    assert pp[0] < pp[1] < pp[2]

    img = first.image
    assert img.shape == (3, 64, 64) and img.dtype == np.float32
    centers = [float(img[1].max()) for img in [s.image for s in sag_samples]]
    assert centers == sorted(centers)


def test_channel_semantics_previous_center_next(synthetic_env, tmp_path):
    env = synthetic_env
    cfg = PreprocessConfig(
        image_size=(32, 32),
        num_samples_per_plane=1,
        boundary_margin=0.5,
        normalization_scope="triplet",
        cache_enabled=False,
    )
    studies = discover_studies(env["headers_csv"], env["series_csv"], cfg)
    prep = StudyPreprocessor(cfg, data_root=env["raw"], cache_dir=None)
    result = prep.process_study(env["study_uid"], studies[env["study_uid"]])
    sag = next(s for s in result.samples if s.metadata["plane_csv"] == "Sagittal")

    assert sag.metadata["physical_positions_mm"][0] < sag.metadata["physical_positions_mm"][1] < sag.metadata["physical_positions_mm"][2]
    prev_mean = float(np.mean(sag.image[0]))
    center_mean = float(np.mean(sag.image[1]))
    next_mean = float(np.mean(sag.image[2]))
    assert prev_mean < center_mean < next_mean
    scopes = {s["scope"] for s in sag.metadata["normalization_stats"]}
    assert scopes == {"triplet"}


def test_determinism_across_runs(synthetic_env, tmp_path):
    env = synthetic_env
    cfg = PreprocessConfig(image_size=(48, 48), cache_enabled=False)
    studies = discover_studies(env["headers_csv"], env["series_csv"], cfg)
    prep = StudyPreprocessor(cfg, data_root=env["raw"], cache_dir=None)
    r1 = prep.process_study(env["study_uid"], studies[env["study_uid"]])
    r2 = prep.process_study(env["study_uid"], studies[env["study_uid"]])
    assert len(r1.samples) == len(r2.samples)
    for a, b in zip(r1.samples, r2.samples):
        assert a.metadata == b.metadata
        assert np.array_equal(a.image, b.image)


def test_caching_reuses_and_config_change_invalidates(synthetic_env, tmp_path):
    env = synthetic_env
    cache_dir = tmp_path / "cache"
    cfg = PreprocessConfig(image_size=(32, 32), cache_enabled=True, cache_dir=cache_dir)
    studies = discover_studies(env["headers_csv"], env["series_csv"], cfg)
    prep = StudyPreprocessor(cfg, data_root=env["raw"], cache_dir=cache_dir)
    r1 = prep.process_study(env["study_uid"], studies[env["study_uid"]])
    files = list(cache_dir.glob("*.npz"))
    assert files
    r2 = StudyPreprocessor(cfg, data_root=env["raw"], cache_dir=cache_dir).process_study(
        env["study_uid"], studies[env["study_uid"]]
    )
    assert len(r2.samples) == len(r1.samples)

    cfg_other = PreprocessConfig(image_size=(64, 64), cache_enabled=True, cache_dir=cache_dir)
    prep_other = StudyPreprocessor(cfg_other, data_root=env["raw"], cache_dir=cache_dir)
    prep_other.process_study(env["study_uid"], studies[env["study_uid"]])
    assert len(list(cache_dir.glob("*.npz"))) > len(files)


def test_insufficient_slices_permissive_skips_strict_raises(tmp_path):
    raw = tmp_path / "raw"
    study_uid = "1.2.826.777.1"
    thin = "1.2.826.888.1"
    for k in range(2):
        write_const_slice(
            raw / "train_series" / thin / f"{k}.dcm",
            study_uid=study_uid,
            series_uid=thin,
            instance_number=k + 1,
            ipp=(float(k), 0.0, 0.0),
            iop=IOP_SAGITTAL,
            value=k,
        )
    series_csv = tmp_path / "train_series.csv"
    pd.DataFrame([{
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": thin,
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
        "Anatomical_Plane": "Sagittal",
    }]).to_csv(series_csv, index=False)
    headers_csv = make_header_cache(tmp_path, raw)

    infos = discover_studies(headers_csv, series_csv)[study_uid]
    qc = validate_series(infos[0], PreprocessConfig())
    assert not qc.usable

    permissive_cfg = PreprocessConfig(planes=("Sagittal",), mode="permissive")
    permissive = StudyPreprocessor(permissive_cfg, data_root=raw, cache_dir=None)
    result = permissive.process_study(study_uid, infos)
    assert result.samples == [] and result.skipped_series

    strict_cfg = PreprocessConfig(planes=("Sagittal",), mode="strict")
    strict = StudyPreprocessor(strict_cfg, data_root=raw, cache_dir=None)
    with pytest.raises(SeriesQCError):
        strict.process_study(study_uid, infos)


def test_dataset_returns_study_batch(synthetic_env):
    env = synthetic_env
    cfg = PreprocessConfig(image_size=(32, 32), cache_enabled=True,
                           cache_dir=env["tmp"] / "cache")
    ds = KneeStudyDataset(
        [env["study_uid"]],
        data_root=env["raw"],
        series_csv=env["series_csv"],
        headers_cache=env["headers_csv"],
        labels_csv=env["labels_csv"],
        cfg=cfg,
        mode="val",
    )
    item = ds[0]
    assert item["images"].shape[1:] == (3, 32, 32)
    assert item["images"].dtype == torch.float32
    n = item["images"].shape[0]
    assert n == len(item["sample_metadata"])
    per_plane: dict[str, int] = {}
    for md in item["sample_metadata"]:
        per_plane[md["plane_csv"]] = per_plane.get(md["plane_csv"], 0) + 1
    assert set(per_plane) == {"Sagittal", "Coronal", "Axial"}
    assert all(0 < v <= cfg.num_samples_per_plane for v in per_plane.values())
    assert item["labels"].shape == (12,)
    assert item["labels"][0].item() == 1.0      # ACL
    assert item["labels"][7].item() == 0.0      # Effusion
    assert torch.isnan(item["labels"][11])
    assert item["study_uid"] == env["study_uid"]
    md = item["sample_metadata"][0]
    assert set(["study_uid", "series_uid", "plane_csv", "slice_indices", "physical_positions_mm"]).issubset(md)


def test_dataset_unlabeled_all_nan(synthetic_env):
    env = synthetic_env
    empty_labels = env["tmp"] / "empty_labels.csv"
    pd.DataFrame([{**{c: np.nan for c in LABELS}, "StudyInstanceUID": env["study_uid"], "Report": "r"}]).to_csv(empty_labels, index=False)
    ds = KneeStudyDataset(
        [env["study_uid"]],
        data_root=env["raw"],
        series_csv=env["series_csv"],
        headers_cache=env["headers_csv"],
        labels_csv=empty_labels,
        cfg=PreprocessConfig(cache_enabled=False),
        mode="inference",
    )
    item = ds[0]
    assert torch.isnan(item["labels"]).all()


def test_unknown_study_raises_keyerror(synthetic_env):
    env = synthetic_env
    with pytest.raises(KeyError):
        KneeStudyDataset(
            ["missing-study"],
            data_root=env["raw"],
            series_csv=env["series_csv"],
            headers_cache=env["headers_csv"],
            labels_csv=env["labels_csv"],
            cfg=PreprocessConfig(cache_enabled=False),
        )


def test_cache_key_depends_on_config_and_sops():
    cfg_a = PreprocessConfig(image_size=(224, 224))
    cfg_b = PreprocessConfig(image_size=(256, 256))
    k1 = series_cache_key(cfg_a, "s1", ["a", "b"])
    k2 = series_cache_key(cfg_b, "s1", ["a", "b"])
    k3 = series_cache_key(cfg_a, "s1", ["a", "b", "c"])
    k4 = series_cache_key(cfg_a, "s2", ["a", "b"])
    assert len({k1, k2, k3, k4}) == 4

