import torch
from tqdm import tqdm
from models.modules.encoder import MaskEncoder, ImageEncoder
from .ddpm import DDPM


class LatentDiffusion(DDPM):
    def __init__(self, timesteps, unet_config, device, **ignore_kwargs):
        super().__init__(timesteps, unet_config, device)
        self.mask_encoder = MaskEncoder().to(self.device)
        if unet_config["use_cross_attn"]:
            assert "context_dim" in unet_config, "context_dim must be specified if using cross attn"
            context_dim = unet_config['context_dim']
            self.img_encoder = ImageEncoder(out_channels=context_dim).to(self.device)
    
    def forward(self, z, t, c={"global": None, "local": None}, noise=None):
        if noise is None:
            noise = torch.randn_like(z)
        z_noisy = self.diffuse_forward(z, t, noise)
        gcond = c['global']
        lcond = c['local']
        if lcond is not None:
            z_noisy = torch.cat([z_noisy, lcond], dim=1)
        z_hat = self.unet(z_noisy, t, gcond)
        return z_hat, noise

    def p_mean_variance(self, x, c, t, clip_denoised: bool = False):
        t_in = t
        gcond = c['global']
        lcond = c['local']
        xc = torch.cat([x, lcond], dim=1)
        model_out = self.unet(xc, t_in, context=gcond)

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=model_out)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_0=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 temperature=1., noise_dropout=0.):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x, c=c, t=t, clip_denoised=clip_denoised)
        model_mean, _, model_log_variance, x_recon = outputs

        noise = torch.randn(x.shape, device=device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, temperature=1., noise_dropout=0.):

        timesteps = self.num_timesteps
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        intermediates = []

        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps)
                        
        temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            img, intermediates = self.p_sample(img, cond, ts,
                                               temperature=temperature[i], clip_denoised=False, noise_dropout=noise_dropout)

        return img, intermediates