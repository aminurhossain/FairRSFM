# Source Code

The FairRSFM implementation will live in the `fairrsfm` Python package. The
package is organized by responsibility so that benchmark construction,
training, mitigation, and evaluation remain independently reusable.

```text
src/fairrsfm/
|-- biomes/       # Ecoregion lookup and 14-to-6 biome mapping
|-- data/         # Dataset metadata, loaders, and preprocessing
|-- evaluation/   # Task, worst-group, calibration, and parity metrics
|-- methods/      # ERM, DBR, BOLP, and GroupDRO
|-- models/       # Frozen RSFM adapters and downstream heads
`-- training/     # Training loops, checkpointing, and seed control
```

The implementation is in preparation. Public modules will follow the protocol
documented in [the benchmark specification](../docs/benchmark.md) and [the
methods reference](../docs/methods.md).
