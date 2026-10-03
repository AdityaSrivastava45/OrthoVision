from pathlib import Path

import pytest

from orthovision.data.config import DataConfig


@pytest.fixture
def data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        dataset_root=tmp_path / "raw",
        output_dir=tmp_path / "out",
        manifest_dir=tmp_path / "out" / "manifests",
        viz_dir=tmp_path / "out" / "viz",
        num_workers=1,
        logging_level="INFO",
        log_file=None,
        plane_min_alignment=0.8,
        plane_min_axis_separation=0.2,
        min_slices_warning=4,
        unusual_spacing_cv=0.15,
        duplicate_position_epsilon_mm=1e-3,
        orientation_epsilon=1e-3,
        project_root=tmp_path,
    )
