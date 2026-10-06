"""Gen-TFM: in-context synthetic tabular data generation with a tabular foundation model."""

from .checkpoint import export_slim_checkpoint, load_pretrained, save_training_checkpoint
from .encoding import Schema, decode_components, mixed_feature_mask, sanitize_mixed_encoded
from .generation import DEFAULT_CALIBRATION, NO_CALIBRATION, Calibration, generate_in_context, generate_zero_context
from .metrics import downstream_accuracy, evaluate_mixed
from .model import GenTFM
from .real_data import build_encoded_table, decode_to_dataframe, encode_dataframe

__all__ = [
    "Schema", "GenTFM", "Calibration", "DEFAULT_CALIBRATION", "NO_CALIBRATION",
    "generate_in_context", "generate_zero_context", "load_pretrained", "save_training_checkpoint",
    "export_slim_checkpoint", "evaluate_mixed", "downstream_accuracy", "encode_dataframe",
    "decode_to_dataframe", "build_encoded_table", "decode_components", "mixed_feature_mask",
    "sanitize_mixed_encoded",
]

__version__ = "0.1.0"
