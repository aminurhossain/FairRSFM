"""Frozen RSFM adapters and downstream heads (Prithvi, SatMAE, DOFA)."""

from fairrsfm.models.prithvi_wrapper import PrithviModelWrapper, get_prithvi_model
from fairrsfm.models.satmae_wrapper import (
    SatMAEFeatureExtractor,
    SatMAEClassifier,
    SatMAEUperNetSegmentationClassifier,
)
from fairrsfm.models.dofa_wrapper import (
    DOFA,
    DOFAFeatureExtractor,
    DOFAClassifier,
    DOFASegmentationClassifier,
    DOFAUperNetSegmentationClassifier,
)
from fairrsfm.models.terrafm_wrapper import (
    TerraFMFeatureExtractor,
    TerraFMClassificationClassifier,
    TerraFMUPerNetSegmentationClassifier,
)
from fairrsfm.models.cloudsen12_models import (
    build_cloudsen12_model,
    get_cloudsen12_band_indices,
)
from fairrsfm.models.lars import LARS

__all__ = [
    "PrithviModelWrapper",
    "get_prithvi_model",
    "SatMAEFeatureExtractor",
    "SatMAEClassifier",
    "SatMAEUperNetSegmentationClassifier",
    "DOFA",
    "DOFAFeatureExtractor",
    "DOFAClassifier",
    "DOFASegmentationClassifier",
    "DOFAUperNetSegmentationClassifier",
    "TerraFMFeatureExtractor",
    "TerraFMClassificationClassifier",
    "TerraFMUPerNetSegmentationClassifier",
    "build_cloudsen12_model",
    "get_cloudsen12_band_indices",
    "LARS",
]
