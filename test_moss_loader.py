"""CPU tests for strict PEFT LoRA fusion: python -m unittest test_moss_loader."""

import unittest
from unittest.mock import patch
import hashlib
import io
import tempfile
from pathlib import Path

import torch

from moss_loader import download_lora, merge_lora


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


if __name__ == "__main__":
    unittest.main()
