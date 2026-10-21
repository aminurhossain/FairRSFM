"""
FairRSFM Biome Taxonomy and Mapping Utilities
=============================================
Defines the 14 RESOLVE terrestrial biomes and their consolidation into the
six macro-groups used as evaluation strata across the benchmark.
"""

from typing import Dict, Optional, Union
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. TAXONOMY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

RAW_BIOMES: Dict[int, str] = {
    1: "Tropical and Subtropical Moist Broadleaf Forests",
    2: "Tropical and Subtropical Dry Broadleaf Forests",
    3: "Tropical and Subtropical Coniferous Forests",
    4: "Temperate Broadleaf and Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests / Taiga",
    7: "Tropical and Subtropical Grasslands, Savannas, and Shrublands",
    8: "Temperate Grasslands, Savannas, and Shrublands",
    9: "Flooded Grasslands and Savannas",
    10: "Montane Grasslands and Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands, and Scrub",
    13: "Deserts and Xeric Shrublands",
    14: "Mangroves",
}

MACRO_GROUPS: Dict[int, str] = {
    1: "Aseasonal High-Biomass",
    2: "High-Amplitude Phenological",
    3: "Transitional Herbaceous and Scrub",
    4: "Cryospheric and Short-Cycle",
    5: "Xeric and Mineralogical",
    6: "Hydrologically Modulated",
}

RAW_TO_MACRO_ID: Dict[int, int] = {
    1: 1,  # Tropical Moist Forests -> Aseasonal High-Biomass
    3: 1,  # Tropical Conifer Forests -> Aseasonal High-Biomass
    5: 1,  # Temperate Conifer Forests -> Aseasonal High-Biomass
    2: 2,  # Tropical Dry Forests -> High-Amplitude Phenological
    4: 2,  # Temperate Broadleaf / Mixed -> High-Amplitude Phenological
    6: 2,  # Boreal Forests / Taiga -> High-Amplitude Phenological
    7: 3,  # Tropical Grasslands / Savannas -> Transitional Herbaceous and Scrub
    8: 3,  # Temperate Grasslands / Savannas -> Transitional Herbaceous and Scrub
    12: 3, # Mediterranean Forests -> Transitional Herbaceous and Scrub
    10: 4, # Montane Grasslands -> Cryospheric and Short-Cycle
    11: 4, # Tundra -> Cryospheric and Short-Cycle
    13: 5, # Deserts and Xeric Shrublands -> Xeric and Mineralogical
    9: 6,  # Flooded Grasslands -> Hydrologically Modulated
    14: 6, # Mangroves -> Hydrologically Modulated
}

RAW_TO_MACRO_NAME: Dict[int, str] = {
    raw_id: MACRO_GROUPS[macro_id]
    for raw_id, macro_id in RAW_TO_MACRO_ID.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOOKUP AND MAPPING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_group(raw_biome_id: Union[int, str]) -> str:
    """
    Map a raw RESOLVE terrestrial biome ID (1-14) to its macro-group name.
    Returns 'Unknown' if the raw biome ID is unmapped or invalid.
    """
    try:
        raw_int = int(raw_biome_id)
        return RAW_TO_MACRO_NAME.get(raw_int, "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


def get_macro_group_id(raw_biome_id: Union[int, str]) -> Optional[int]:
    """
    Map a raw RESOLVE terrestrial biome ID (1-14) to its macro-group ID (1-6).
    Returns None if unmapped.
    """
    try:
        raw_int = int(raw_biome_id)
        return RAW_TO_MACRO_ID.get(raw_int, None)
    except (ValueError, TypeError):
        return None


def load_biome_mapping(mapping_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load a dataset biome mapping JSON file and return a dictionary mapping
    sample ID (as string) to macro-group name.
    """
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"Biome mapping file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Some mappings are dictionaries keyed by sample ID; others are lists of records
    mapping = {}
    if isinstance(records, dict):
        for k, v in records.items():
            if isinstance(v, dict):
                code = v.get("biome_code") or v.get("biome_id") or v.get("biome")
            else:
                code = v
            mapping[str(k)] = get_macro_group(code)
    elif isinstance(records, list):
        for r in records:
            sid = r.get("geobench_id") or r.get("sample_id") or r.get("id") or r.get("patch_id")
            code = r.get("biome_code") or r.get("biome_id") or r.get("biome")
            if sid is not None:
                mapping[str(sid)] = get_macro_group(code)

    return mapping
