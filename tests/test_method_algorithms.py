"""Numerical protocol and checkpoint tests for shared training algorithms."""

import copy

import pytest
import torch

from fairrsfm.methods.dbr import DynamicBiomeReweighter
from fairrsfm.methods.group_dro import GroupDROLossTracker
from fairrsfm.methods.bolp import compute_bolp_projection, PrithviFinalLayerBOLP
from fairrsfm.methods.task_loss import per_sample_task_loss


def test_inverse_frequency_initialization_and_neutral_unknown():
    dbr = DynamicBiomeReweighter(["a", "a", "a", "b", "Unknown"])
    assert dbr.weights.tolist() == pytest.approx([0.5, 1.5])
    assert dbr.weights_for_groups(["Unknown", "unseen"]).tolist() == [1, 1]


def test_group_loss_is_epoch_mean_and_beta_smooths_losses():
    dbr = DynamicBiomeReweighter(["a", "b"], beta=0.9)
    dbr.observe(torch.tensor([1., 3., 4.]), ["a", "a", "b"])
    dbr.observe(torch.tensor([8.]), ["b"])
    dbr.finish_epoch(0)
    assert dbr.ema_loss.tolist() == pytest.approx([2, 6])
    expected = torch.tensor([2 + 1e-6, 6 + 1e-6]).sqrt()
    assert dbr.weights.tolist() == pytest.approx((expected / expected.mean()).tolist())
    dbr.observe(torch.tensor([12., 2.]), ["a", "b"])
    dbr.finish_epoch(1)
    assert dbr.ema_loss.tolist() == pytest.approx([3, 5.6])


def test_unequal_batches_do_not_bias_group_mean():
    dbr = DynamicBiomeReweighter(["a"])
    dbr.observe(torch.tensor([1., 1., 1.]), ["a"] * 3)
    dbr.observe(torch.tensor([9.]), ["a"])
    dbr.finish_epoch(0)
    assert dbr.ema_loss.item() == pytest.approx(3)


def test_absent_group_retains_ema_and_unknown_does_not_contribute():
    dbr = DynamicBiomeReweighter(["a", "b"])
    dbr.observe(torch.tensor([2., 4.]), ["a", "b"])
    dbr.finish_epoch(0)
    dbr.observe(torch.tensor([6., 1000.]), ["a", "Unknown"])
    dbr.finish_epoch(1)
    assert dbr.ema_loss.tolist() == pytest.approx([2.4, 4])


def test_partial_epoch_checkpoint_resume_matches_uninterrupted_update():
    dbr = DynamicBiomeReweighter(["a", "a", "b"])
    dbr.observe(torch.tensor([2., 6.]), ["a", "b"])
    dbr.finish_epoch(0)
    dbr.observe(torch.tensor([3.]), ["a"])
    resumed = DynamicBiomeReweighter(["a", "a", "b"])
    resumed.load_state_dict(copy.deepcopy(dbr.state_dict()))
    for instance in [dbr, resumed]:
        instance.observe(torch.tensor([5., 1.]), ["a", "b"])
        instance.finish_epoch(1)
    torch.testing.assert_close(resumed.weights, dbr.weights)
    torch.testing.assert_close(resumed.ema_loss, dbr.ema_loss)
    with pytest.raises(ValueError, match="already updated"):
        resumed.finish_epoch(1)


def test_resume_rejects_changed_method_parameters():
    state = DynamicBiomeReweighter(["a", "b"], beta=0.9).state_dict()
    with pytest.raises(ValueError, match="hyperparameters"):
        DynamicBiomeReweighter(["a", "b"], beta=0.5).load_state_dict(state)


def test_groupdro_uses_updated_probabilities_as_official_code_does():
    tracker = GroupDROLossTracker(["a", "b", "c"], group_lr=0.01)
    losses = torch.tensor([1., 3., 5.], requires_grad=True)
    result = tracker(losses, ["a", "a", "b"])
    expected_q = torch.softmax(torch.tensor([0.02, 0.05, 0.]), 0)
    torch.testing.assert_close(tracker.q, expected_q)
    torch.testing.assert_close(result, expected_q[0] * 2 + expected_q[1] * 5)
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor(
        [expected_q[0] / 2, expected_q[0] / 2, expected_q[1]]))


def test_segmentation_loss_is_mean_of_valid_pixels_per_patch():
    logits = torch.tensor([[[[3., -3.]], [[-3., 3.]]],
                           [[[3., -3.]], [[-3., 3.]]]])
    target = torch.tensor([[[0, -1]], [[0, 0]]])
    result = per_sample_task_loss(logits, target, "mmearth")
    expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1, reduction="none")
    torch.testing.assert_close(result, torch.stack([expected[0, 0, 0], expected[1].mean()]))
    target[0] = -1
    assert torch.isnan(per_sample_task_loss(logits, target, "mmearth")[0])


def test_bolp_uses_unweighted_group_mean_and_projects_every_spatial_position():
    embeddings = torch.tensor([[0., 0., 0.], [0., 0., 0.], [2., 0., 0.]])
    projector = compute_bolp_projection(embeddings, ["a", "a", "b"])
    torch.testing.assert_close(projector.mu0, torch.tensor([1., 0., 0.]))
    tokens = torch.tensor([[[3., 4., 5.], [6., 7., 8.]]])
    expected = torch.tensor([[[0., 4., 5.], [0., 7., 8.]]])
    torch.testing.assert_close(projector(tokens, ["a"]), expected)
    maps = tokens.transpose(1, 2).unsqueeze(-1)
    torch.testing.assert_close(projector(maps, ["a"]), expected.transpose(1, 2).unsqueeze(-1))


def test_prithvi_multilayer_bolp_fits_and_transforms_only_final_layer():
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return [images + 100, images]

    class BackboneAndHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

        def forward(self, images):
            self.features = self.encoder(images)
            return self.features[-1].sum(1)

    model = BackboneAndHead()
    bolp = PrithviFinalLayerBOLP(tile_lookup=lambda sid, mapping: mapping.get(sid, "Unknown"))
    mapping = {"a": "A", "b": "B"}
    images = torch.tensor([[[0., 2.], [0., 4.]], [[2., 2.], [2., 4.]]])
    loader = [(images, torch.zeros(2), ["a", "b"])]
    bolp.fit(model, loader, mapping)
    assert bolp.projector.k == 1
    torch.testing.assert_close(bolp.projector.mu0, torch.tensor([1., 3.]))
    bolp.forward_with_bolp(model, images, ["a", "b"], mapping)
    torch.testing.assert_close(model.features[0], images + 100)
    torch.testing.assert_close(model.features[-1][..., 0], torch.zeros(2, 2))
    assert not model.encoder._forward_hooks
    assert "projector.Pk" in bolp.state_dict()


def test_generic_probe_keeps_decoder_layers_and_projects_final_tokens():
    from fairrsfm.training.lightning_module import FairRSFMProbe

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return [x + 100, x]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

        def forward(self, x):
            self.features = self.encoder(x)
            return self.features[-1].mean(1)

    projector = compute_bolp_projection(torch.tensor([[0., 0.], [2., 0.]]), ["A", "B"])
    model = Model()
    probe = FairRSFMProbe(model, method="bolp", biome_map={"a": "A"}, bolp_projector=projector)
    tokens = torch.tensor([[[3., 2.], [5., 4.]]])
    torch.testing.assert_close(probe(tokens, ["a"]), torch.tensor([[0., 3.]]))
    torch.testing.assert_close(model.features[0], tokens + 100)
    torch.testing.assert_close(model.features[-1][..., 0], torch.zeros(1, 2))
    assert not model.encoder._forward_hooks
