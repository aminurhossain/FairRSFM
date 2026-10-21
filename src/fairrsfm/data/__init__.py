"""Dataset metadata, loaders, and preprocessing for FairRSFM benchmark."""

from fairrsfm.data.geobench import GeoBenchDataset
from fairrsfm.data.mmearth import (
    MMEarthDynamicWorldSegmentationDataset,
    MMEarthDynamicWorldSegmentationDataModule,
)

# Convenience alias
MMEarthDataModule = MMEarthDynamicWorldSegmentationDataModule

__all__ = [
    "GeoBenchDataset",
    "MMEarthDataModule",
    "MMEarthDynamicWorldSegmentationDataset",
    "MMEarthDynamicWorldSegmentationDataModule",
]
