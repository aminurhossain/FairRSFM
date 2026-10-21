"""Frozen RSFM adapters and downstream heads (Prithvi, SatMAE, DOFA)."""

from fairrsfm.models.prithvi_wrapper import PrithviModelWrapper, get_prithvi_model
from fairrsfm.models.satmae_wrapper import SatMAEFeatureExtractor, SatMAEClassifier
from fairrsfm.models.dofa_wrapper import (
    DOFA,
    DOFAFeatureExtractor,
    DOFAClassifier,
    DOFASegmentationClassifier,
    DOFAUperNetSegmentationClassifier,
)
from fairrsfm.models.lars import LARS

__all__ = [
    "PrithviModelWrapper",
    "get_prithvi_model",
    "SatMAEFeatureExtractor",
    "SatMAEClassifier",
    "DOFA",
    "DOFAFeatureExtractor",
    "DOFAClassifier",
    "DOFASegmentationClassifier",
    "DOFAUperNetSegmentationClassifier",
    "LARS",
]
