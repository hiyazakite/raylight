"""Tests for Lightricks FSDP sharding helpers and registry wiring."""
import importlib
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


# ---------------------------------------------------------------------------
# Dummy model fixtures
# ---------------------------------------------------------------------------

class _DummyConnector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.transformer_1d_blocks = nn.ModuleList([
            nn.Linear(width, width),
            nn.Linear(width, width),
        ])


class _DummyLTXAVDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, 4)
        self.transformer_blocks = nn.ModuleList([
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        ])
        self.video_embeddings_connector = _DummyConnector(4)
        self.audio_embeddings_connector = _DummyConnector(4)
        self.output_proj = nn.Linear(4, 4)


@pytest.fixture
def fsdp_module():
    from raylight.diffusion_models.lightricks import fsdp as lightricks_fsdp
    return lightricks_fsdp


@pytest.fixture
def ltxav_model():
    return _DummyLTXAVDiffusionModel()


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestLightricksFSDPHelpers:
    def test_iter_shard_targets_ltxav(self, fsdp_module, ltxav_model):
        targets = fsdp_module._iter_shard_targets(ltxav_model)
        prefixes = [prefix for prefix, _ in targets]
        assert prefixes == ["transformer_blocks"]

    def test_iter_prebake_targets_ltxav(self, fsdp_module, ltxav_model):
        targets = fsdp_module._iter_prebake_targets(ltxav_model)
        prefixes = [prefix for prefix, _ in targets]
        assert prefixes == [
            "video_embeddings_connector",
            "audio_embeddings_connector",
        ]

    def test_iter_shard_targets_ltxv(self, fsdp_module):
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])
        targets = fsdp_module._iter_shard_targets(model)
        assert len(targets) == 1
        assert targets[0][0] == "transformer_blocks"

    def test_iter_shard_targets_legacy_connector(self, fsdp_module):
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])
        model.embeddings_connector = _DummyConnector(4)

        targets = fsdp_module._iter_shard_targets(model)
        prebake_targets = fsdp_module._iter_prebake_targets(model)
        assert [p for p, _ in targets] == ["transformer_blocks"]
        assert [p for p, _ in prebake_targets] == ["embeddings_connector"]

    def test_collect_ignored_params_excludes_sharded_blocks(self, fsdp_module, ltxav_model):
        shard_prefixes = ["transformer_blocks"]
        ignored_params = fsdp_module._collect_ignored_params(ltxav_model, shard_prefixes)
        ignored_names = {
            name
            for name, param in ltxav_model.named_parameters()
            if param in ignored_params
        }

        assert "input_proj.weight" in ignored_names
        assert "input_proj.bias" in ignored_names
        assert "output_proj.weight" in ignored_names
        assert "output_proj.bias" in ignored_names
        assert "transformer_blocks.0.weight" not in ignored_names
        assert "video_embeddings_connector.transformer_1d_blocks.0.weight" in ignored_names
        assert "audio_embeddings_connector.transformer_1d_blocks.0.weight" in ignored_names

    def test_format_shard_target_summary(self, fsdp_module):
        summary = fsdp_module._format_shard_target_summary(
            [
                ("transformer_blocks", [object(), object()]),
                ("video_embeddings_connector.transformer_1d_blocks", [object()]),
            ]
        )
        assert summary == "transformer_blocks=2, video_embeddings_connector.transformer_1d_blocks=1"

    def test_format_prebake_target_summary(self, fsdp_module):
        summary = fsdp_module._format_prebake_target_summary(
            [
                ("video_embeddings_connector", _DummyConnector(4)),
                ("audio_embeddings_connector", _DummyConnector(4)),
            ]
        )
        assert summary == "video_embeddings_connector=4p, audio_embeddings_connector=4p"

    def test_find_unmatched_ltx_patch_keys_all_handled(self, fsdp_module):
        """All patch keys fall under handled prefixes — nothing unmatched."""
        patcher = types.SimpleNamespace(
            patches={
                "diffusion_model.transformer_blocks.0.weight": object(),
                "diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.weight": object(),
                "diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.weight": object(),
                "diffusion_model.embeddings_connector.transformer_1d_blocks.0.weight": object(),
                "diffusion_model.audio_embeddings_connector.unexpected.weight": object(),
                "diffusion_model.other.weight": object(),
            }
        )

        unmatched = fsdp_module._find_unmatched_ltx_patch_keys(
            patcher,
            [
                "transformer_blocks",
                "video_embeddings_connector",
                "audio_embeddings_connector",
                "embeddings_connector",
            ],
        )

        # audio_embeddings_connector.unexpected.weight IS handled because
        # the prefix "audio_embeddings_connector" covers all children.
        # "other.weight" is not in relevant_markers, so also not reported.
        assert unmatched == []

    def test_find_unmatched_ltx_patch_keys_with_unhandled(self, fsdp_module):
        """Patch key under a relevant prefix but NOT in handled_prefixes."""
        patcher = types.SimpleNamespace(
            patches={
                "diffusion_model.transformer_blocks.0.weight": object(),
                "diffusion_model.audio_embeddings_connector.layer.weight": object(),
            }
        )

        # Only transformer_blocks is handled — audio_embeddings_connector is not.
        unmatched = fsdp_module._find_unmatched_ltx_patch_keys(
            patcher,
            ["transformer_blocks"],
        )

        assert unmatched == ["diffusion_model.audio_embeddings_connector.layer.weight"]

    def test_find_unmatched_none_patcher(self, fsdp_module):
        assert fsdp_module._find_unmatched_ltx_patch_keys(None, ["transformer_blocks"]) == []


# ---------------------------------------------------------------------------
# Registry wiring tests
# ---------------------------------------------------------------------------

class TestFSDPRegistryLTXWiring:
    @pytest.fixture(autouse=True)
    def _setup_fake_comfy(self):
        # Save comfy_dist submodules that may get re-imported with our fake comfy
        saved_comfy_dist = {}
        for key in list(sys.modules):
            if key.startswith("raylight.comfy_dist"):
                saved_comfy_dist[key] = sys.modules.pop(key)

        fake_comfy = types.ModuleType("comfy")
        fake_comfy.__path__ = []
        fake_model_base = types.SimpleNamespace(
            LTXAV=type("LTXAV", (), {}),
            LTXV=type("LTXV", (), {}),
        )
        fake_comfy.model_base = fake_model_base

        saved_comfy = {}
        for name in list(sys.modules):
            if name == "comfy" or name.startswith("comfy."):
                saved_comfy[name] = sys.modules.pop(name)

        sys.modules["comfy"] = fake_comfy
        sys.modules["comfy.model_base"] = fake_model_base

        # Bypass raylight.comfy_dist.__init__ entirely — import fsdp_registry
        # as a standalone module so its heavy comfy-dependent siblings don't load.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "raylight.comfy_dist.fsdp_registry",
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                         "src", "raylight", "comfy_dist", "fsdp_registry.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["raylight.comfy_dist.fsdp_registry"] = mod
        spec.loader.exec_module(mod)

        self.fake_model_base = fake_model_base
        self.registry_module = mod

        yield

        # Teardown: restore original module state
        for key in list(sys.modules):
            if key.startswith("raylight.comfy_dist"):
                sys.modules.pop(key, None)
        for key, m in saved_comfy_dist.items():
            sys.modules[key] = m
        for name in list(sys.modules):
            if name == "comfy" or name.startswith("comfy."):
                sys.modules.pop(name, None)
        for name, old in saved_comfy.items():
            sys.modules[name] = old

    def test_ltx_types_registered(self):
        registry = self.registry_module.FSDPShardRegistry._REGISTRY
        assert self.fake_model_base.LTXAV in registry
        assert self.fake_model_base.LTXV in registry

    def test_wrap_dispatches_ltxav(self):
        sentinel = object()
        original = self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV]

        try:
            self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV] = (
                lambda model, sd, cpu_offload, patcher=None: sentinel
            )
            wrapped = self.registry_module.FSDPShardRegistry.wrap(self.fake_model_base.LTXAV())
            assert wrapped is sentinel
        finally:
            self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV] = original
