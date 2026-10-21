"""Ecoregion lookup and 14-to-6 biome macro-group mapping."""

from fairrsfm.biomes.taxonomy import (
    RAW_BIOMES,
    MACRO_GROUPS,
    RAW_TO_MACRO_ID,
    RAW_TO_MACRO_NAME,
    get_macro_group,
    get_macro_group_id,
    load_biome_mapping,
)

__all__ = [
    "RAW_BIOMES",
    "MACRO_GROUPS",
    "RAW_TO_MACRO_ID",
    "RAW_TO_MACRO_NAME",
    "get_macro_group",
    "get_macro_group_id",
    "load_biome_mapping",
]
