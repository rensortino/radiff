from datasets import RGMasksDataset
from datasets.utils import get_data_loader
from trainers.t_ldm import LDMTrainer
from utils.model import read_model_config
from utils.parser import get_parser, parse_args


def main():
    parser = get_parser()
    args = parse_args(parser)
    assert args["unet_config"] is not None, "Specify a config file for the UNet"
    assert args["ae_config"] is not None, "Specify a config file for the Autoencoder"

    ae_kwargs = read_model_config(args["ae_config"])
    ae_kwargs.update({"ae_ckpt": args["ae_ckpt"]})

    unet_kwargs = read_model_config(args["unet_config"])

    dset_train = RGMasksDataset(args["data_root"], args["dataset"], img_size=128)
    dset_test = RGMasksDataset(args["data_root"], args["dataset"], img_size=128)

    logger_kwargs = {
        "output_dir": args["run_dir"],
        "run_name": args["run_name"],
        "on_wandb": args["on_wandb"],
        "wandb_entity": args["wandb_entity"],
        "wandb_project": args["wandb_project"],
    }

    train_loader = get_data_loader(dset_train, args["batch_size"], split="train")
    test_loader = get_data_loader(dset_test, args["batch_size"], split="test")

    default_kwargs = {
        "device": args["device"],
        "epochs": args["epochs"],
        "save_freq": args["save_freq"],
        "log_freq": args["log_freq"],
        "debug": args["debug"],
        "train_loader": train_loader,
        "test_loader": test_loader,
        "logger_kwargs": logger_kwargs,
    }

    ldm_kwargs = {
        "timesteps": args["timesteps"],
        "unet_config": unet_kwargs,
        "l_simple_weight": args["l_simple_weight"],
        "original_elbo_weight": args["original_elbo_weight"],
        "logvar_init": args["logvar_init"],
        "ddim_steps": args["ddim_steps"],
        "ddim_eta": args["ddim_eta"],
        "device": args["device"],
    }

    # Setup trainer
    trainer = LDMTrainer(
        args["lr"], default_kwargs, ldm_kwargs, ae_kwargs, args["resume"]
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
