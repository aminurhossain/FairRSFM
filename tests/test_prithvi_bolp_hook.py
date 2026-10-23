"""The checkpoint evaluator must preserve the encoder's token tensor shape."""

import unittest

import torch
from torch import nn

from fairrsfm.evaluation.prithvi.bolp import attach_classification_bolp


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class PrithviBOLPHookTests(unittest.TestCase):
    def test_tokens_keep_shape_and_use_group_residualization(self):
        model = TokenModel()
        checkpoint = {"bolp_biome_index": {"forest": 0},
                      "bolp_skip_residualization": False}
        state = {"projector.mu0": torch.tensor([0.0, 0.0]),
                 "projector.Pk": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                 "projector.mu_table": torch.tensor([[1.0, 0.0]])}
        attach_classification_bolp(model, checkpoint, state, torch.device("cpu"))
        model.bolp_biome_ids = ["forest"]
        tokens = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
        result = model(tokens)
        self.assertEqual(result.shape, tokens.shape)
        torch.testing.assert_close(result, torch.tensor([[[1.0, -1.0], [3.0, 1.0]]]))
        self.assertEqual(list(model.state_dict()), [])

    def test_missing_biome_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            attach_classification_bolp(TokenModel(), {}, {
                "projector.mu0": torch.zeros(2),
                "projector.Pk": torch.eye(2),
                "projector.mu_table": torch.zeros(1, 2),
            }, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
