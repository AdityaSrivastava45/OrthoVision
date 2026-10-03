from pathlib import Path

from pydicom.uid import generate_uid

from orthovision.data.catalog import KneeMRIDataset
from orthovision.data.config import DataConfig
from orthovision.data.visualize import visualize_study
from tests.synth_dicom import IOP_SAGITTAL, write_mr_slice


def test_visualize_writes_png(tmp_path: Path, data_cfg: DataConfig):
    raw = data_cfg.dataset_root
    study, series = generate_uid(), generate_uid()
    for i in range(5):
        write_mr_slice(
            raw / f"{i}.dcm",
            study_uid=study,
            series_uid=series,
            instance_number=i + 1,
            ipp=(float(i), 0, 0),
            iop=IOP_SAGITTAL,
        )
    ds = KneeMRIDataset(data_cfg)
    ds.scan()
    dest = tmp_path / "viz"
    paths = visualize_study(ds.get_study(study), dest, n_slices=3)
    assert paths
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 0
