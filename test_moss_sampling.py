"""CPU checks for ComfyUI flow scheduling, stochastic sampling and SLG boundaries."""

import unittest

import torch

from comfy_moss_sampling import sample_er_sde, simple_sigmas, uncond_skip_active


class MossSamplingTests(unittest.TestCase):
    def test_simple_schedule_uses_shifted_training_grid(self):
        sigmas = simple_sigmas(25)
        self.assertEqual(len(sigmas), 26)
        self.assertEqual(sigmas[0], 1)
        self.assertEqual(sigmas[-1], 0)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        self.assertAlmostEqual(float(sigmas[1]), 5 * .96 / (1 + 4 * .96), places=6)

    def test_sigma_interval_skips_first_thirteen_simple_steps(self):
        active = [uncond_skip_active(s) for s in simple_sigmas(25)[:-1]]
        self.assertEqual(active, [True] * 13 + [False] * 12)
        self.assertTrue(uncond_skip_active(.84))
        self.assertFalse(uncond_skip_active(.82))

    def test_constant_denoiser_has_exact_terminal_output(self):
        calls = []
        x = torch.ones(1, 1, 2, 2)

        def model(value, sigma, **kwargs):
            calls.append(float(sigma[0]))
            return torch.full_like(value, .25)

        result = sample_er_sde(model, x, simple_sigmas(25), disable=True)
        self.assertEqual(len(calls), 25)
        self.assertLess(calls[0], 1)  # ER-SDE's logSNR safety offset.
        torch.testing.assert_close(result, torch.full_like(x, .25), rtol=0, atol=0)

    def test_seeded_sde_noise_is_reproducible_and_finite(self):
        x = torch.linspace(-1, 1, 16).reshape(1, 1, 4, 4)

        def model(value, sigma, **kwargs):
            return .2 * value + sigma.reshape(-1, 1, 1, 1) ** 2

        a = sample_er_sde(model, x.clone(), simple_sigmas(25), extra_args={"seed": 42}, disable=True)
        b = sample_er_sde(model, x.clone(), simple_sigmas(25), extra_args={"seed": 42}, disable=True)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
