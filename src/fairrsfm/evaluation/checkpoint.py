"""Validate that a fine-tuned prediction head was actually restored."""


def load_finetuned_state(model, state_dict):
    """Load a slim checkpoint, allowing omitted frozen backbone weights only.

    Lightning training strips frozen encoder/backbone tensors from some runs.
    An absent or unexpected task head is an error: accepting it with
    ``strict=False`` would report test metrics from random weights.
    """
    result = model.load_state_dict(state_dict, strict=False)
    frozen_prefixes = ("encoder.", "backbone.")
    missing = [key for key in result.missing_keys if not key.startswith(frozen_prefixes)]
    unexpected = [key for key in result.unexpected_keys if not key.startswith(frozen_prefixes)]
    if missing or unexpected:
        raise ValueError(
            "Checkpoint architecture does not match evaluator: "
            f"missing non-backbone keys={missing[:8]}, "
            f"unexpected non-backbone keys={unexpected[:8]}"
        )
    return result
