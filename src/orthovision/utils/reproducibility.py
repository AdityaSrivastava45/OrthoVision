"""Reproducibility utilities for deterministic training."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

log = logging.getLogger("orthovision.utils.reproducibility")


@dataclass
class DeterministicConfig:
    """Configuration for deterministic behavior."""
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = False  # cudnn.benchmark


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, enable deterministic algorithms (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Disable cudnn benchmarking for determinism
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Set environment variable for cuBLAS
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log.info(f"Deterministic mode enabled with seed {seed}")
    else:
        # For performance, allow cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        log.info(f"Seeds set to {seed} (non-deterministic mode)")

    # Log the seed
    log.info(f"Random seed: {seed}")


def get_deterministic_config(config: dict[str, Any] | None = None) -> DeterministicConfig:
    """Get deterministic config from configuration dict."""
    if config is None:
        config = {}
    return DeterministicConfig(
        seed=config.get("seed", 42),
        deterministic=config.get("deterministic", False),
        benchmark=config.get("cudnn_benchmark", False),
    )


def set_deterministic_mode(config: DeterministicConfig) -> None:
    """Apply deterministic configuration."""
    set_seed(config.seed, config.deterministic)
    if config.deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = config.benchmark


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Worker initialization function for DataLoader."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """Create PyTorch generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_system_info() -> dict[str, Any]:
    """Log system information for reproducibility."""
    info = {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
            info[f"gpu_{i}_memory"] = torch.cuda.get_device_properties(i).total_memory

    log.info(f"System info: {info}")
    return info