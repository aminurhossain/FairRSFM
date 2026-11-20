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

    def test_new_protocol_projects_all_tokens_and_preserves_earlier_layers(self):
        model = TokenModel()
        class Features(nn.Module):
            def forward(self, x):
                return [x + 10, x]
        model.encoder = Features()
        checkpoint = {"bolp_biome_index": {"forest": 0},
                      "bolp_skip_residualization": False,
                      "hyper_parameters": {"method_protocol": {"bolp_transform": "tokenwise_v1"}}}
        state = {"projector.mu0": torch.zeros(2),
                 "projector.Pk": torch.diag(torch.tensor([1., 0.])),
                 "projector.mu_table": torch.tensor([[1., 0.]])}
        attach_classification_bolp(model, checkpoint, state, torch.device("cpu"))
        tokens = torch.tensor([[[2., 3.], [4., 5.]]])
        model.bolp_biome_ids = ["forest"]
        early, final = model(tokens)
        torch.testing.assert_close(early, tokens + 10)
        torch.testing.assert_close(final, torch.tensor([[[1., 0.], [3., 0.]]]))
        # Unknown inference groups still receive the global projection.
        model.bolp_biome_ids = None
        torch.testing.assert_close(model(tokens)[-1], torch.tensor([[[2., 0.], [4., 0.]]]))


if __name__ == "__main__":
    unittest.main()
