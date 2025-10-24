"""Utility functions for agents"""
import math
import numpy as np
from typing import Any

def _sanitize_float_values(data: Any) -> Any:
    """
    Recursively sanitize float values to ensure JSON compliance.
    Replaces inf, -inf, and NaN with safe values.
    """
    if isinstance(data, dict):
        return {key: _sanitize_float_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_sanitize_float_values(item) for item in data]
    elif isinstance(data, (np.floating, float)):
        if math.isnan(data):
            return 0.0
        elif math.isinf(data):
            return 1e6 if data > 0 else -1e6
        else:
            return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif hasattr(data, 'item'):  # numpy scalar
        return data.item()
    else:
        return data
