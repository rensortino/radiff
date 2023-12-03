import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from models.ddpm import DDPM
from models.modules.ddim import DDIMSampler
from trainers.t_base import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, lr, default_kwargs, ddpm_kwargs, resume=None):
        super().__init__(**default_kwargs)
        self.ddpm = DDPM(**ddpm_kwargs).to(self.device)
        self.loss = torch.nn.MSELoss().to(self.device)
        self.opt = AdamW(self.ddpm.parameters(), lr)
        self.init_checkpoint(resume)
        self.l_simple_weight = ddpm_kwargs["l_simple_weight"]
        self.original_elbo_weight = ddpm_kwargs["original_elbo_weight"]
        self.logvar = torch.full(
            fill_value=ddpm_kwargs["logvar_init"],
            size=(self.ddpm.num_timesteps,),
            device=self.device,
        )
        self.epoch_losses = ["loss", "loss_simple", "loss_vlb"]

        self.ddim_steps = ddpm_kwargs["ddim_steps"]
        self.ddim_eta = ddpm_kwargs["ddim_eta"]

        self.gen_masks = ddpm_kwargs["unet_config"]["gen_masks"]

    def _make_ckpt(self, epoch, loss):
        return {
            "epoch": epoch,
            "loss": loss,
            "model": self.ddpm.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)

        self.ddpm.load_state_dict(checkpoint["model"], strict=False)
        self.start_epoch = checkpoint["epoch"]
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt"])

    def run_step(self, batch):
        x = batch
        x = x.to(self.device)
        t = torch.randint(
            0, self.ddpm.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        out, noise = self.ddpm(x, t)

        target = noise

        loss_dict = {}

        loss_simple = F.mse_loss(out, target, reduction="none").mean([1, 2, 3])
        loss_dict.update({f"loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        # if self.learn_logvar:
        #     loss_dict.update({f'loss_gamma': loss.mean()})
        #     loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = F.mse_loss(out, target, reduction="none").mean(dim=(1, 2, 3))
        loss_vlb = (self.ddpm.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"loss": loss})

        return loss, loss_dict

    def train_one_epoch(self):
        epoch_loss_dict = {k: 0 for k in self.epoch_losses}
        self.ddpm.train()
        for batch in tqdm(self.train_loader):
            loss, loss_dict = self.run_step(batch)
            epoch_loss_dict = {
                k: v + loss_dict[k].item() for k, v in epoch_loss_dict.items()
            }

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.debug:
                break

        # Aggregating and logging metrics
        epoch_loss_dict = {
            k: v / len(self.train_loader) for k, v in epoch_loss_dict.items()
        }
        return epoch_loss_dict

    def eval_one_epoch(self):
        epoch_loss_dict = {k: 0 for k in self.epoch_losses}
        self.ddpm.eval()
        samples = {}
        shape = [
            self.test_loader.batch_size,
            self.ddpm.unet.in_channels,
            self.ddpm.unet.image_size,
            self.ddpm.unet.image_size,
        ]
        bs = shape[0]
        c = {"global": None, "local": None}

        batch = next(iter(self.test_loader))
        gt = batch
        _, loss_dict = self.run_step(batch)
        epoch_loss_dict = {k: v + loss_dict[k] for k, v in epoch_loss_dict.items()}

        ddim = DDIMSampler(self.ddpm)
        with torch.no_grad():
            # generated = self.ddpm.p_sample_loop(
            #     shape)
            generated, _ = ddim.sample(
                self.ddim_steps, bs, shape[1:], c, eta=self.ddim_eta
            )
        generated = generated.clamp(0, 1).round()
        # generated = mask_to_rgb(generated)
        samples = {"gt": gt, "gen": generated}

        # Aggregating and logging metrics
        epoch_loss_dict = {k: v for k, v in epoch_loss_dict.items()}
        return epoch_loss_dict, samples
