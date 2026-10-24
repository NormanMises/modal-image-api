# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI (Comfy-Org), commit bc5f880dbdd43e426077f42ccec53bb577ffee9f.
# Source: comfy/k_diffusion/sampling.py and comfy/samplers.py.
# License: LICENSE-ComfyUI.txt.
# Changes: specialize SNR helpers and model sampling to MOSS CONST flow;
# accept a denoiser callable without ComfyUI's model-patcher object graph.
"""ComfyUI-compatible ER-SDE solver and simple schedule for MOSS-Image."""

import torch
from tqdm.auto import trange

UPSTREAM_COMMIT = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"


class FlowSampling:
    def __init__(self, shift=5.0):
        self.shift = shift
        t = torch.arange(1, 1001, dtype=torch.float32) / 1000
        self.sigmas = shift * t / (1 + (shift - 1) * t)

    def percent_to_sigma(self, percent):
        if percent <= 0:
            return 1.0
        if percent >= 1:
            return 0.0
        t = 1.0 - percent
        return self.shift * t / (1 + (self.shift - 1) * t)


def simple_sigmas(steps=25):
    sampling = FlowSampling()
    stride = len(sampling.sigmas) / steps
    values = [float(sampling.sigmas[-(1 + int(i * stride))]) for i in range(steps)]
    return torch.tensor(values + [0.0], dtype=torch.float32)


def uncond_skip_active(sigma):
    sampling = FlowSampling()
    return sampling.percent_to_sigma(0.5) <= float(sigma) <= sampling.percent_to_sigma(0)


def sigma_to_half_log_snr(sigma, model_sampling):
    return sigma.logit().neg()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    if len(sigmas) > 1 and sigmas[0] >= 1:
        sigmas = sigmas.clone()
        sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas


def default_noise_sampler(x, seed=None):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1

        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, noise_scaler=None, max_stage=3):
    """Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3). arXiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_noise = s_noise  # MOSS CONST flow noise_scale = 1.0
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(x):
        return x * ((x ** 0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    model_sampling = FlowSampling()
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x
