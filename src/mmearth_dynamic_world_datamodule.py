"""Compatibility import for legacy training scripts.

The maintained implementation lives in fairrsfm.mmearth_dynamic_world_datamodule.
"""

from fairrsfm.mmearth_dynamic_world_datamodule import (  # noqa: F401
    DYNAMIC_WORLD_CLASSES,
    MMEarthDynamicWorldSegmentationDataModule,
    MMEarthDynamicWorldSegmentationDataset,
    find_one,
)
