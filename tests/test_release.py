import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ISM_diffusion.config import get_model_config
from ISM_diffusion.data import StructuralDataset, load_conditions, read_split
from ISM_diffusion.models.diffusion.ddim import DDIMSampler
from ISM_diffusion.modules.attention import CrossAttention
from ISM_diffusion.modules.diffusionmodules.util import checkpoint
from ISM_diffusion.training import PaperControlLDM, TrainableEMA, checkpoint_weights, freeze_for_paper
from ISM_diffusion.util import get_clip_text_embedding


class ReleaseChecks(unittest.TestCase):
    def test_condition_contract_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, horiz=np.zeros((64, 128)), fault=np.ones((64, 128)))
            data = load_conditions(path)
            self.assertEqual(data["horiz"].dtype, np.float32)
            self.assertEqual(data["fault"].sum(), 64 * 128)
            for horiz in [np.zeros((65, 128)), np.full((64, 128), np.nan), np.full((64, 128), 2.)]:
                np.savez(path, horiz=horiz, fault=np.zeros((64, 128)))
                with self.assertRaises(ValueError):
                    load_conditions(path)

    def test_training_target_mapping_and_validation_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            rgt = np.broadcast_to(np.linspace(0, 50, 512, dtype=np.float32)[:, None], (512, 512))
            np.savez(path, rgt=rgt, fault=np.zeros_like(rgt))
            dataset = StructuralDataset(directory, [path.name], training=False)
            first, second = dataset[0], dataset[0]
            self.assertEqual(first["jpg"].shape, (3, 512, 512))
            self.assertTrue(torch.equal(first["jpg"][0], first["jpg"][2]))
            self.assertTrue(torch.equal(first["horiz"], second["horiz"]))
            self.assertTrue((first["fault"] == -1).all())
            self.assertTrue((first["horiz"] > -1).any())

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            for groups in [dict(train=["a"], val=["a"], test=["c"]),
                           dict(train=["../a"], val=["b"], test=["c"])]:
                path.write_text(json.dumps(groups))
                with self.assertRaises(ValueError):
                    read_split(path)

    def test_packaged_embedding_and_independent_configs(self):
        embedding = get_clip_text_embedding()
        self.assertEqual(embedding.shape, (1, 77, 1024))
        self.assertTrue(np.isfinite(embedding).all())
        embedding[:] = 0
        self.assertTrue(np.any(get_clip_text_embedding()))
        config = get_model_config()
        config["params"]["timesteps"] = 2
        self.assertEqual(get_model_config()["params"]["timesteps"], 1000)

    def test_sdpa_matches_reference_attention(self):
        torch.manual_seed(1)
        attention = CrossAttention(32, context_dim=48, heads=4, dim_head=8)
        x, context = torch.randn(2, 7, 32), torch.randn(2, 5, 48)
        q, k, v = [projection(t).view(2, -1, 4, 8).transpose(1, 2)
                   for projection, t in [(attention.to_q, x), (attention.to_k, context), (attention.to_v, context)]]
        expected = ((q @ k.transpose(-1, -2) / 8**0.5).softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 32)
        torch.testing.assert_close(attention(x, context), attention.to_out(expected), atol=1e-6, rtol=1e-5)

    def test_checkpoint_backward_with_optional_context(self):
        layer = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        checkpoint(lambda t, context: layer(t), (x, None), layer.parameters(), True).sum().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_ddim_buffers_follow_cpu_model(self):
        sampler = DDIMSampler(SimpleNamespace(num_timesteps=1000, device=torch.device("cpu")))
        sampler.register_buffer("test", torch.ones(2))
        self.assertEqual(sampler.test.device.type, "cpu")

    def test_freeze_scope(self):
        model = torch.nn.Module()
        model.control_fault = torch.nn.Linear(2, 2)
        model.control_horiz = torch.nn.Linear(2, 2)
        model.first_stage_model = torch.nn.Linear(2, 2)
        model.model = torch.nn.Module()
        model.model.diffusion_model = torch.nn.Module()
        for name in ("input_blocks", "middle_block", "time_embed", "output_blocks", "out"):
            setattr(model.model.diffusion_model, name, torch.nn.Linear(2, 2))
        freeze_for_paper(model)
        for name, parameter in model.named_parameters():
            expected = name.startswith(("control_fault.", "control_horiz.",
                                        "model.diffusion_model.output_blocks.", "model.diffusion_model.out."))
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertFalse(model.first_stage_model.training)

    def test_ema_update_resume_swap_and_selection(self):
        layer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2)
        ema = TrainableEMA(decay=0.5)
        trainer = SimpleNamespace(global_step=0)
        ema.on_fit_start(trainer, layer)
        with torch.no_grad():
            layer.weight.fill_(4)
        trainer.global_step = 1
        ema.on_train_batch_end(trainer, layer, None, None, 0)
        self.assertEqual(ema.shadow["weight"].item(), 3)
        ema.on_train_batch_end(trainer, layer, None, None, 0)
        self.assertEqual(ema.shadow["weight"].item(), 3)
        ema.on_validation_start(trainer, layer)
        self.assertEqual(layer.weight.item(), 3)
        ema.on_validation_end(trainer, layer)
        self.assertEqual(layer.weight.item(), 4)
        resumed = TrainableEMA(decay=0.5)
        resumed.load_state_dict(ema.state_dict())
        resumed.on_fit_start(trainer, layer)
        trainer.global_step = 2
        resumed.on_train_batch_end(trainer, layer, None, None, 0)
        self.assertEqual(resumed.shadow["weight"].item(), 3.5)
        saved = dict(state_dict=layer.state_dict(), callbacks={ema.state_key: resumed.state_dict()})
        self.assertEqual(checkpoint_weights(saved, use_ema=True)["weight"].item(), 3.5)
        self.assertEqual(checkpoint_weights(saved)["weight"].item(), 4)
        with self.assertRaises(ValueError):
            checkpoint_weights(dict(state_dict=layer.state_dict()), use_ema=True)

    def test_actual_model_training_and_lightning_resume(self):
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        torch.set_num_threads(2)
        config = get_model_config()["params"]
        for key in ("unet_config", "control_stage_config"):
            config[key]["params"].update(model_channels=32, channel_mult=[1, 2],
                                         num_res_blocks=1, attention_resolutions=[],
                                         num_head_channels=32, context_dim=32)
        config["first_stage_config"]["params"]["ddconfig"].update(ch=32, ch_mult=[1, 2, 2, 2])

        def make_model():
            model = PaperControlLDM(**config)
            model.clip_txt = torch.zeros(1, 77, 32)
            model.learning_rate = 1e-5
            return model

        batch = dict(jpg=torch.rand(3, 64, 64) * 2 - 1,
                     fault=torch.zeros(1, 64, 64), horiz=torch.zeros(1, 64, 64))
        loader = DataLoader([batch, batch], batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            model, ema = make_model(), TrainableEMA()
            options = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                           enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                           default_root_dir=directory)
            trainer = pl.Trainer(max_steps=1, callbacks=[ema], **options)
            trainer.fit(model, loader)
            self.assertEqual(ema.last_step, 1)
            self.assertTrue(all(torch.isfinite(v).all() for v in ema.shadow.values()))
            self.assertTrue(all(p.grad is None for p in model.first_stage_model.parameters()))
            checkpoint_path = Path(directory) / "last.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            saved = torch.load(checkpoint_path, weights_only=False)
            self.assertIn("ISMTrainableEMA", saved["callbacks"])
            self.assertTrue(saved["optimizer_states"])
            resumed_model, resumed_ema = make_model(), TrainableEMA()
            resumed = pl.Trainer(max_steps=2, callbacks=[resumed_ema], **options)
            resumed.fit(resumed_model, loader, ckpt_path=checkpoint_path)
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(resumed_ema.last_step, 2)


if __name__ == "__main__":
    unittest.main()
