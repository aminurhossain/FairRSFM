"""Convert metric results to standards-compliant JSON values."""

import math

import numpy as np


def json_safe(value):
    """Replace undefined metrics with null while preserving nested structure."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
