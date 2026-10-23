"""Execute an existing training script with explicit smoke-only runtime limits.

No production configuration is changed: use through train.py --smoke-test.
BOLP fitting and callback evaluation see at most eight real batches; Lightning
runs two train/validation/test batches, one epoch, with checkpointing enabled.
"""

import itertools
import runpy
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader


def main():
    script, *args = sys.argv[1:]
    original_init = pl.Trainer.__init__
    original_iter = DataLoader.__iter__

    def smoke_init(self, *positional, **kwargs):
        kwargs.update(max_epochs=1, limit_train_batches=2, limit_val_batches=2,
                      limit_test_batches=2, num_sanity_val_steps=0)
        original_init(self, *positional, **kwargs)

    def smoke_iter(self):
        return itertools.islice(original_iter(self), 8)

    pl.Trainer.__init__ = smoke_init
    DataLoader.__iter__ = smoke_iter
    sys.argv = [script, *args]
    print("SMOKE ONLY: one epoch, two Lightning batches, eight batches for BOLP/callbacks", flush=True)
    runpy.run_path(str(Path(script).resolve()), run_name="__main__")
    if torch.cuda.is_available():
        print(f"SMOKE PASSED; peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
