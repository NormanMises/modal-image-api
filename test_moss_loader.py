"""CPU tests for strict PEFT LoRA fusion: python -m unittest test_moss_loader."""

import unittest
from unittest.mock import patch
import hashlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from moss_loader import MOSS_NEGATIVE_PROMPT, MossBackend, download_lora, merge_lora


class MergeLoraTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.proj = torch.nn.Linear(3, 2, bias=False)
        self.model.proj.weight.data.zero_()
        self.a = torch.tensor([[1.0, 2.0, 3.0]])
        self.b = torch.tensor([[2.0], [4.0]])
        self.state = {
            "diffusion_model.proj.lora_A.weight": self.a,
            "diffusion_model.proj.lora_B.weight": self.b,
        }

    def test_peft_strength_one(self):
        self.assertEqual(merge_lora(self.model, self.state), 1)
        torch.testing.assert_close(self.model.proj.weight, self.b @ self.a)

    def test_preserves_base_weight_and_dtype(self):
        self.model.to(dtype=torch.bfloat16)
        self.model.proj.weight.data.fill_(1)
        merge_lora(self.model, self.state)
        self.assertEqual(self.model.proj.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(self.model.proj.weight, (1 + self.b @ self.a).bfloat16())

    def test_unknown_key_does_not_partially_merge(self):
        self.state["unknown.weight"] = self.a
        with self.assertRaises(ValueError):
            merge_lora(self.model, self.state)
        self.assertEqual(self.model.proj.weight.count_nonzero().item(), 0)

    def test_missing_pair_is_rejected(self):
        del self.state["diffusion_model.proj.lora_B.weight"]
        with self.assertRaises(ValueError):
            merge_lora(self.model, self.state)

    def test_base_shape_mismatch_is_rejected(self):
        self.model.proj = torch.nn.Linear(4, 2, bias=False)
        with self.assertRaises(ValueError):
            merge_lora(self.model, self.state)


class DownloadLoraTests(unittest.TestCase):
    def test_verified_download_is_reused(self):
        payload = b"test model weights"
        with tempfile.TemporaryDirectory() as cache:
            with patch("moss_loader.LORA_SHA256", hashlib.sha256(payload).hexdigest()):
                with patch("moss_loader.urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                    path = download_lora(cache)
                    self.assertEqual(path.read_bytes(), payload)
                    self.assertEqual(download_lora(cache), path)
                    request.assert_called_once()
            self.assertEqual(list(Path(cache).iterdir()), [path])

    def test_bad_checksum_is_not_cached(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch("moss_loader.urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaises(ValueError):
                    download_lora(cache)
            self.assertEqual(list(Path(cache).iterdir()), [])


class MossFlowAdapterTests(unittest.TestCase):
    def test_comfy_sigmas_convert_to_official_training_timesteps(self):
        from PIL import Image

        calls = []

        class Transformer:
            config = SimpleNamespace(latent_channels=4)

            def __call__(self, x, timestep, embeds, mask, **kwargs):
                calls.append((float(timestep[0]), kwargs))
                return SimpleNamespace(sample=torch.zeros_like(x))

        encode = Mock(return_value=(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)))
        backend = MossBackend.__new__(MossBackend)
        backend.pipe = SimpleNamespace(
            _execution_device=torch.device("cpu"),
            transformer=Transformer(),
            scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
            encode_prompt=encode,
            maybe_free_model_hooks=Mock(),
        )
        backend._decode_latents = Mock(return_value=Image.new("RGB", (16, 16)))
        result = backend.generate("test prompt", 16, 16, seed=42)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(calls), 50)
        self.assertGreater(calls[0][0], 999)
        self.assertLess(calls[0][0], 1000)
        self.assertEqual(sum(bool(k.get("skip_layers")) for _, k in calls), 13)
        self.assertEqual(encode.call_args_list[1].args, (MOSS_NEGATIVE_PROMPT,))
        self.assertNotIn("nsfw", MOSS_NEGATIVE_PROMPT.lower())
        backend.pipe.maybe_free_model_hooks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
