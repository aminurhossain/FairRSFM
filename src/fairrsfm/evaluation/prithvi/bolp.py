"""Restore the Prithvi classification BOLP transform used during training."""

import torch
from torch import nn


class ClassificationBOLPProjector(nn.Module):
    def __init__(self, checkpoint: dict, state_dict: dict):
        super().__init__()
        required = ("projector.mu0", "projector.Pk", "projector.mu_table")
        missing = [name for name in required if name not in state_dict]
        if missing:
            raise ValueError(f"BOLP checkpoint is missing projector buffers: {missing}")
        if "bolp_biome_index" not in checkpoint or "bolp_skip_residualization" not in checkpoint:
            raise ValueError("BOLP checkpoint lacks biome index or residualization metadata")
        self.biome_index = checkpoint["bolp_biome_index"]
        self.skip_residualization = checkpoint["bolp_skip_residualization"]
        protocol = checkpoint.get("hyper_parameters", {}).get("method_protocol", {})
        self.tokenwise = protocol.get("bolp_transform") == "tokenwise_v1"
        self.register_buffer("mu0", state_dict["projector.mu0"])
        self.register_buffer("Pk", state_dict["projector.Pk"])
        self.register_buffer("mu_table", state_dict["projector.mu_table"])

    def forward(self, z: torch.Tensor, biome_ids: list[str]) -> torch.Tensor:
        spatial = z.ndim == 4
        z_tilde = z.movedim(1, -1).clone() if spatial else z.clone()
        if not self.skip_residualization:
            for i, biome in enumerate(biome_ids):
                index = self.biome_index.get(biome)
                if index is not None:
                    z_tilde[i] = z_tilde[i] - self.mu_table[index] + self.mu0
        result = z_tilde @ self.Pk.T
        return result.movedim(-1, 1) if spatial else result


def attach_classification_bolp(model: nn.Module, checkpoint: dict, state_dict: dict,
                               device: torch.device) -> None:
    """Match the encoder hook in the classification training scripts."""
    projector = ClassificationBOLPProjector(checkpoint, state_dict).to(device)
    # Keep projector outside the task model's state_dict: its buffers have
    # already been restored from the checkpoint separately.
    object.__setattr__(model, "bolp_projector", projector)
    model.bolp_biome_ids = None

    def hook_fn(module, inputs, output):
        biome_ids = model.bolp_biome_ids
        if biome_ids is None and not projector.tokenwise:
            return output
        if biome_ids is None:
            last = output[-1] if isinstance(output, (list, tuple)) else output
            biome_ids = ["Unknown"] * last.shape[0]
        last = output[-1] if isinstance(output, (list, tuple)) else output
        if last.dim() == 3 and not projector.tokenwise:
            z_avg = last.mean(dim=1)
            z_hat = projector(z_avg, biome_ids)
            transformed = last + (z_hat - z_avg).unsqueeze(1)
        else:
            transformed = projector(last, biome_ids)
        if isinstance(output, (list, tuple)):
            out_list = list(output)
            out_list[-1] = transformed
            return type(output)(out_list)
        return transformed

    model.encoder.register_forward_hook(hook_fn)
