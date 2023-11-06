import torch
from torch.optim import Adam
from tqdm import tqdm

from models.vqgan import VQModel
from trainers.t_base import BaseTrainer


class VQVAETrainer(BaseTrainer):
    def __init__(self, lr, default_kwargs, vqvae_kwargs, resume=None):
        super().__init__(**default_kwargs)
        # setup vq model

        self.vqmodel = VQModel(
            **vqvae_kwargs['params']).to(self.device)

        # setup optimizers
        self.opt_ae = Adam(self.vqmodel.parameters(),
                           lr=lr, betas=(0.5, 0.9))

        self.codebook_weight = 1.0

        self.init_checkpoint(resume)

    def _make_ckpt(self, epoch, loss):
        return {
            "epoch": epoch,
            "loss": loss,
            "vqmodel": self.vqmodel.state_dict(),
            "opt": self.opt_ae.state_dict(),
        }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if 'vqmodel' in checkpoint:
            self.vqmodel.load_state_dict(checkpoint['vqmodel'], strict=False)
            self.start_epoch = checkpoint['epoch']
        else:
            self.vqmodel.load_state_dict(checkpoint)

    def train_one_epoch(self):
        # Set model training mode
        self.vqmodel.train()

        loss_epoch = {}

        for batch in tqdm(self.train_loader):
            # self.global_step += 1

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
            loss_epoch[key] = sum(
                loss_epoch[key]) / len(loss_epoch[key])

        return loss_epoch
    
    def eval_one_epoch(self):
        # Set model eval mode
        self.vqmodel.eval()

        ae_loss_epoch = {}

        for batch in tqdm(self.test_loader):
            # Eval autoencoder
            _, ae_loss_dict, _ = self.run_step(batch, split="eval")

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
            ae_loss_epoch[key] = sum(
                ae_loss_epoch[key]) / len(ae_loss_epoch[key])

        # Perform evaluation step on a signle batch
        sample_batch = next(iter(self.test_loader))
        _, _, samples = self.run_step(sample_batch, split="eval")

        return ae_loss_epoch, samples

    def run_step(self, batch, split="train"):
        x = batch
        x = x.to(self.device)
        xrec, qloss = self.vqmodel(x)

        rec_loss = torch.abs(x.contiguous() -
                             xrec.contiguous()).mean()
        loss = rec_loss + self.codebook_weight * qloss.mean()

        loss_dict = {f"rec_loss": rec_loss,
                     f"qloss": qloss.mean(),
                     f"loss": loss}

        return loss, loss_dict, {"gt": x, "rec": xrec.detach()}
