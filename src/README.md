# Source Code

The FairRSFM implementation lives in the `fairrsfm` Python package. The
package is organized by responsibility so that benchmark construction,
training, mitigation, and evaluation remain independently reusable.

```text
src/fairrsfm/
|-- biomes/       # Ecoregion lookup and 14-to-6 biome mapping (taxonomy.py)
|-- data/         # Dataset metadata, loaders, and preprocessing (geobench.py, mmearth.py)
|-- evaluation/   # Task, worst-group, calibration, and parity metrics (metrics.py, evaluate.py)
|-- methods/      # ERM, DBR, BOLP, and GroupDRO
|-- models/       # Frozen RSFM adapters and downstream heads (Prithvi, SatMAE, DOFA)
`-- training/     # Training loops, LightningModule, and seed control
```

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

## Quick API Example

```python
import fairrsfm
from fairrsfm.biomes import get_macro_group, RAW_BIOMES
from fairrsfm.methods import DynamicBiomeReweighter, BOLPProjector, GroupDROLossTracker
from fairrsfm.models import get_prithvi_model, SatMAEClassifier, DOFAClassifier
from fairrsfm.evaluation import compute_fairness_summary

# Map any RESOLVE terrestrial biome code to one of 6 macro-groups:
macro_group = get_macro_group(1) # 'Aseasonal High-Biomass'
```

For running training runs across the 48 benchmark configurations, use the launcher in `experiments/`:

```bash
python experiments/train.py --model satmae --dataset bigearthnet --method groupdro --seed 42
```
