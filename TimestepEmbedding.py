# This file is adapted from https://github.com/explainingai-code/DDPM-Pytorch
# Original license: MIT
# Copyright (c) 2020 Original Author

import torch

class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
        
    def sample_prev_timestep(self, x0, xt, t):
        if t == 0:
            return x0,x0

        device = xt.device

        beta_t = self.betas[t].to(device)
        alpha_t = self.alphas[t].to(device)
        sqrt_alpha_t = self.sqrt_alphas[t].to(device)
        alpha_bar_t = self.alpha_cum_prod[t].to(device)
        alpha_bar_prev = self.alpha_cum_prod[t - 1].to(device)
        sqrt_alpha_bar_prev = self.sqrt_alpha_cum_prod[t - 1].to(device)

        coef_x0 = (sqrt_alpha_bar_prev * beta_t) / (1 - alpha_bar_t)
        coef_xt = (sqrt_alpha_t * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)

        mean = coef_x0 * x0 + coef_xt * xt
        posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        sigma = torch.sqrt(posterior_variance)
        noise = torch.randn_like(xt)

        return mean + sigma * noise, x0
