import torch
from torch.optim import Adam
from tqdm import tqdm

from models.autoencoder import AutoencoderKL
from trainers.t_base import BaseTrainer
from models.modules.lpips import LPIPS
from torch import nn


class VAETrainer(BaseTrainer):
    def __init__(self, lr, default_kwargs, ae_kwargs, resume=None):
        super().__init__(**default_kwargs)

        # Define loss weights
        self.kl_weight = ae_kwargs["kl_weight"]
        self.pixel_weight = ae_kwargs["pixelloss_weight"]
        self.perceptual_weight = ae_kwargs["perceptual_weight"]

        # setup AE
        self.vae = AutoencoderKL(**ae_kwargs["params"]).to(self.device)

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_loss = self.perceptual_loss.to(self.device)
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * ae_kwargs["logvar_init"])

        # setup optimizers
        self.opt_ae = Adam(self.vae.parameters(), lr=lr, betas=(0.5, 0.9))

        self.init_checkpoint(resume)

    def _make_ckpt(self, epoch, loss):
        return {
            "epoch": epoch,
            "loss": loss,
            "vae": self.vae.state_dict(),
            "opt": self.opt_ae.state_dict(),
        }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if "vae" in checkpoint:
            self.vae.load_state_dict(checkpoint["vae"], strict=False)
            self.start_epoch = checkpoint["epoch"]
        else:
            self.vae.load_state_dict(checkpoint)

    def train_one_epoch(self):
        # Set model training mode
        self.vae.train()

        loss_epoch = {}

        for batch in tqdm(self.train_loader):
            # Train autoencoder
            ae_loss, ae_loss_dict, _ = self.run_step(batch)

            self.opt_ae.zero_grad()
            ae_loss.backward()
            self.opt_ae.step()

            if loss_epoch == {}:
                for key in ae_loss_dict.keys():
                    loss_epoch[key] = [ae_loss_dict[key].item()]
            else:
                for key in ae_loss_dict.keys():
                    loss_epoch[key] += [ae_loss_dict[key].item()]
            if self.debug:
                break

        for key in loss_epoch.keys():
            loss_epoch[key] = sum(loss_epoch[key]) / len(loss_epoch[key])

        return loss_epoch

    def eval_one_epoch(self):
        # Set model eval mode
        self.vae.eval()

        ae_loss_epoch = {}

        for batch in tqdm(self.test_loader):
            # Eval autoencoder
            _, ae_loss_dict, _ = self.run_step(batch)

            if ae_loss_epoch == {}:
                for key in ae_loss_dict.keys():
                    ae_loss_epoch[key] = [ae_loss_dict[key].item()]
            else:
                for key in ae_loss_dict.keys():
                    ae_loss_epoch[key] += [ae_loss_dict[key].item()]
            if self.debug:
                break

        # Compute loss avg
        for key in ae_loss_epoch.keys():
            ae_loss_epoch[key] = sum(ae_loss_epoch[key]) / len(ae_loss_epoch[key])

        # Perform evaluation step on a signle batch
        sample_batch = next(iter(self.test_loader))
        _, _, samples = self.run_step(sample_batch)

        return ae_loss_epoch, samples

    def run_step(self, batch):
        x = batch
        x = x.to(self.device)
        xrec, posteriors = self.vae(x)

        rec_loss = torch.abs(x.contiguous() - xrec.contiguous()).mean()
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = rec_loss + self.kl_weight * kl_loss

        loss_dict = {f"rec_loss": rec_loss, f"kl_loss": kl_loss, f"loss": loss}

        return loss, loss_dict, {"gt": x, "rec": xrec.detach()}
