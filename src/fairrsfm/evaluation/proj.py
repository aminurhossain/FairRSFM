"""Locate a usable PROJ database before importing geospatial dependencies."""

import os
from pathlib import Path


def ensure_proj_data() -> None:
    current = Path(os.environ.get("PROJ_DATA", ""))
    if (current / "proj.db").is_file():
        return
    try:
        import pyproj
    except ImportError:
        return
    data_dir = Path(pyproj.datadir.get_data_dir())
    if (data_dir / "proj.db").is_file():
        os.environ["PROJ_DATA"] = str(data_dir)
        os.environ["PROJ_LIB"] = str(data_dir)
