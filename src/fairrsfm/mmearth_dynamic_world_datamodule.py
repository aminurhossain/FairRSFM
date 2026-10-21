"""
Compatibility shim module for MMEarth Dynamic World DataModule.
Forwards all symbols directly to fairrsfm.data.mmearth.
"""

from fairrsfm.data.mmearth import *
from fairrsfm.data.mmearth import (
    DYNAMIC_WORLD_CLASSES,
    DEFAULT_DATA_ROOT,
    MMEarthDynamicWorldSegmentationDataModule,
    MMEarthDynamicWorldSegmentationDataset,
    find_one,
    validate_indices,
)
