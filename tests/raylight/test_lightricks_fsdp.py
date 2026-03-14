import importlib
import os
import sys
import types
import unittest

import torch
import torch.nn as nn


sys.path.append("/root/ComfyUI/custom_nodes/raylight/src")


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


class TestLightricksFSDPHelpers(unittest.TestCase):
    def setUp(self):
        from raylight.diffusion_models.lightricks import fsdp as lightricks_fsdp

        self.fsdp = lightricks_fsdp
        self.model = _DummyLTXAVDiffusionModel()

    def test_iter_shard_targets_ltxav(self):
        targets = self.fsdp._iter_shard_targets(self.model)
        prefixes = [prefix for prefix, _ in targets]

        self.assertEqual(prefixes, ["transformer_blocks"])

    def test_iter_prebake_targets_ltxav(self):
        targets = self.fsdp._iter_prebake_targets(self.model)
        prefixes = [prefix for prefix, _ in targets]

        self.assertEqual(
            prefixes,
            [
                "video_embeddings_connector",
                "audio_embeddings_connector",
            ],
        )

    def test_iter_shard_targets_ltxv(self):
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])

        targets = self.fsdp._iter_shard_targets(model)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0][0], "transformer_blocks")

    def test_iter_shard_targets_legacy_connector(self):
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])
        model.embeddings_connector = _DummyConnector(4)

        targets = self.fsdp._iter_shard_targets(model)
        prebake_targets = self.fsdp._iter_prebake_targets(model)
        prefixes = [prefix for prefix, _ in targets]
        prebake_prefixes = [prefix for prefix, _ in prebake_targets]

        self.assertEqual(prefixes, ["transformer_blocks"])
        self.assertEqual(prebake_prefixes, ["embeddings_connector"])

    def test_collect_ignored_params_excludes_sharded_blocks(self):
        shard_prefixes = [
            "transformer_blocks",
        ]

        ignored_params = self.fsdp._collect_ignored_params(self.model, shard_prefixes)
        ignored_names = {
            name
            for name, param in self.model.named_parameters()
            if param in ignored_params
        }

        self.assertIn("input_proj.weight", ignored_names)
        self.assertIn("input_proj.bias", ignored_names)
        self.assertIn("output_proj.weight", ignored_names)
        self.assertIn("output_proj.bias", ignored_names)
        self.assertNotIn("transformer_blocks.0.weight", ignored_names)
        self.assertIn("video_embeddings_connector.transformer_1d_blocks.0.weight", ignored_names)
        self.assertIn("audio_embeddings_connector.transformer_1d_blocks.0.weight", ignored_names)

    def test_format_shard_target_summary(self):
        summary = self.fsdp._format_shard_target_summary(
            [
                ("transformer_blocks", [object(), object()]),
                ("video_embeddings_connector.transformer_1d_blocks", [object()]),
            ]
        )

        self.assertEqual(
            summary,
            "transformer_blocks=2, video_embeddings_connector.transformer_1d_blocks=1",
        )

    def test_format_prebake_target_summary(self):
        summary = self.fsdp._format_prebake_target_summary(
            [
                ("video_embeddings_connector", _DummyConnector(4)),
                ("audio_embeddings_connector", _DummyConnector(4)),
            ]
        )

        self.assertEqual(
            summary,
            "video_embeddings_connector=4p, audio_embeddings_connector=4p",
        )

    def test_find_unmatched_ltx_patch_keys(self):
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

        unmatched = self.fsdp._find_unmatched_ltx_patch_keys(
            patcher,
            [
                "transformer_blocks",
                "video_embeddings_connector",
                "audio_embeddings_connector",
                "embeddings_connector",
            ],
        )

        self.assertEqual(
            unmatched,
            ["diffusion_model.audio_embeddings_connector.unexpected.weight"],
        )


class TestFSDPRegistryLTXWiring(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("raylight.comfy_dist.fsdp_registry", None)

        fake_comfy = types.ModuleType("comfy")
        fake_model_base = types.SimpleNamespace(
            LTXAV=type("LTXAV", (), {}),
            LTXV=type("LTXV", (), {}),
        )
        fake_comfy.model_base = fake_model_base

        self._old_comfy = sys.modules.get("comfy")
        self._old_comfy_model_base = sys.modules.get("comfy.model_base")
        sys.modules["comfy"] = fake_comfy
        sys.modules["comfy.model_base"] = fake_model_base

        self.fake_model_base = fake_model_base
        self.registry_module = importlib.import_module("raylight.comfy_dist.fsdp_registry")

    def tearDown(self):
        sys.modules.pop("raylight.comfy_dist.fsdp_registry", None)
        if self._old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = self._old_comfy

        if self._old_comfy_model_base is None:
            sys.modules.pop("comfy.model_base", None)
        else:
            sys.modules["comfy.model_base"] = self._old_comfy_model_base

    def test_ltx_types_registered(self):
        registry = self.registry_module.FSDPShardRegistry._REGISTRY

        self.assertIn(self.fake_model_base.LTXAV, registry)
        self.assertIn(self.fake_model_base.LTXV, registry)

    def test_wrap_dispatches_ltxav(self):
        sentinel = object()
        original = self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV]

        try:
            self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV] = (
                lambda model, sd, cpu_offload, patcher=None: sentinel
            )
            wrapped = self.registry_module.FSDPShardRegistry.wrap(self.fake_model_base.LTXAV())
            self.assertIs(wrapped, sentinel)
        finally:
            self.registry_module.FSDPShardRegistry._REGISTRY[self.fake_model_base.LTXAV] = original


if __name__ == "__main__":
    unittest.main()