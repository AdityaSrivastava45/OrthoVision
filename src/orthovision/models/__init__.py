"""OrthoVision models package."""

from orthovision.models.dinov2_encoder import DINOv2Encoder, create_dinov2_encoder
from orthovision.models.pooling import MeanPooling, PoolingModule
from orthovision.models.baseline import DINOv2Baseline
from orthovision.models.attention_mil import (
    LabelSpecificAttentionPooling,
    LabelSpecificClassifier,
    AttentionMILConfig,
    create_attention_mil_pooling,
    create_label_classifier,
    check_attention_normalization,
    check_attention_nan,
    check_variable_N_support,
    check_masking_behavior,
    check_provenance_mapping,
    run_attention_sanity_checks,
)

__all__ = [
    "DINOv2Encoder",
    "create_dinov2_encoder",
    "MeanPooling",
    "PoolingModule",
    "DINOv2Baseline",
    "LabelSpecificAttentionPooling",
    "LabelSpecificClassifier",
    "AttentionMILConfig",
    "create_attention_mil_pooling",
    "create_label_classifier",
    "check_attention_normalization",
    "check_attention_nan",
    "check_variable_N_support",
    "check_masking_behavior",
    "check_provenance_mapping",
    "run_attention_sanity_checks",
]