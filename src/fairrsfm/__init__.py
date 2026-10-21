"""
FairRSFM: Biome-Aware Benchmark & Debiasing Framework for Remote Sensing Foundation Models
===========================================================================================
"""

__version__ = "0.1.0"
__author__ = "FairRSFM Authors"

import os
import sys
from pathlib import Path

# Ensure PROJ geospatial database is discovered by rasterio/pyproj/terratorch
if "PROJ_LIB" not in os.environ or not (Path(os.environ["PROJ_LIB"]) / "proj.db").is_file():
    candidates = []
    try:
        import pyproj.datadir
        candidates.append(Path(pyproj.datadir.get_data_dir()))
    except Exception:
        pass
    candidates.extend([
        Path(sys.prefix) / "share" / "proj",
        Path("/home/aminur/anaconda3/share/proj"),
        Path("/usr/share/proj"),
    ])
    for candidate in candidates:
        if (candidate / "proj.db").is_file():
            os.environ["PROJ_LIB"] = str(candidate)
            os.environ["PROJ_DATA"] = str(candidate)
            break

from fairrsfm import biomes, data, evaluation, methods, models, training

__all__ = [
    "biomes",
    "data",
    "evaluation",
    "methods",
    "models",
    "training",
]
