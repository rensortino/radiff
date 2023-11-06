from utils.parser import get_parser, parse_args
from utils.model import read_model_config
from trainers.t_vqvae import VQVAETrainer
from datasets.radiogalaxy import RGDataset
from trainers.t_vae import VAETrainer
from datasets.utils import get_data_loader

def main():
    parser = get_parser()
    args = parse_args(parser)
    assert args['ae_config'] is not None, "Specify a config file for the Autoencoder"

    ae_kwargs = read_model_config(args["ae_config"])
    ae_type = ae_kwargs['type']

    # Load dataset
    dset_train = RGDataset(args['data_root'], args['dataset'], img_size=128)
    dset_test = RGDataset(args['data_root'], args['dataset'], img_size=128)

    logger_kwargs = {
        "output_dir": args["run_dir"],
        "run_name": args["run_name"],
        "on_wandb": args["on_wandb"],
        "wandb_entity": args["wandb_entity"],
        "wandb_project": args["wandb_project"]
   }
    
    train_loader = get_data_loader(dset_train, args['batch_size'], split="train")
    test_loader = get_data_loader(dset_test, args['batch_size'], split="test")

    default_kwargs = {
        "device": args["device"],
        "epochs": args["epochs"],
        "save_freq": args["save_freq"],
        "debug": args["debug"],
        "train_loader": train_loader,
        "test_loader": test_loader,
        "logger_kwargs": logger_kwargs,
    }

    # Setup trainer
    if ae_type == "kl":
        ae_kwargs.update({"logvar_init": args["logvar_init"]})
        trainer = VAETrainer(args['lr'], default_kwargs, ae_kwargs)
    elif ae_type == "vq":
        trainer = VQVAETrainer(args['lr'], default_kwargs, ae_kwargs)
    else:
        raise ValueError(f"Unknown Autoencoder type: {ae_type}")

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
    