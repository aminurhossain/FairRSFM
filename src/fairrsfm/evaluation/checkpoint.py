"""Checkpoint safeguards shared by standalone evaluators."""


def reject_unified_protocol_checkpoint(checkpoint):
    """Prevent legacy evaluators from silently changing a new method's model.

    Unified runs write ``test_selected.json`` during training. That report uses
    the exact model adapter, data preprocessing, BOLP hook, thresholds, and
    shared metric contract that produced the checkpoint. The standalone
    evaluator scripts predate this protocol and remain available for historical
    checkpoints only.
    """
    protocol = checkpoint.get("hyper_parameters", {}).get("method_protocol")
    if protocol:
        method = protocol.get("method", "unknown")
        raise ValueError(
            f"This is a unified {method} checkpoint. Read the run's "
            "test_selected.json (or test_selected_worst.json) produced by the "
            "training adapter. The standalone evaluator supports historical "
            "checkpoints only because it cannot safely reconstruct every new "
            "adapter and BOLP final-layer transform."
        )


def load_finetuned_state(model, state_dict):
    """Load a slim checkpoint, allowing omitted frozen backbone weights only.

    Lightning training strips frozen encoder/backbone tensors from some runs.
    An absent or unexpected task head is an error: accepting it with
    ``strict=False`` would report test metrics from random weights.
    """
    # These buffers govern training objectives and have no prediction role.
    training_prefixes = ("method_protocol.", "group_dro.")
    training_keys = {"running_group_errors", "running_group_loss", "running_errors", "group_weights", "q"}
    prediction_state = {k: v for k, v in state_dict.items()
                        if not k.startswith(training_prefixes) and k not in training_keys}
    result = model.load_state_dict(prediction_state, strict=False)
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
