import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # Model config
    parser.add_argument('--ae-config')
    parser.add_argument('--unet-config')
    # Dataset options
    parser.add_argument('--data-root', type=str, default="data/rg-dataset/data")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--workers', type=int, default=-1,
                        help='-1 for <batch size> threads, 0 for main thread, >0 for background threads')
    # Training options
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    # LDM options
    parser.add_argument('--ae-ckpt')
    parser.add_argument('--ldm-ckpt')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--logvar-init', type=float, default=0.)
    parser.add_argument('--l-simple-weight', type=float, default=1.)
    parser.add_argument('--original-elbo-weight', type=float, default=0.)
    parser.add_argument('--ddim-steps', type=int, default=100)
    parser.add_argument('--ddim-eta', type=float, default=1.0)
    # Logging options
    parser.add_argument('--run-dir', type=str, default='runs/')
    parser.add_argument('--run-name', type=str, default='placeholder')
    parser.add_argument('--on-wandb', type=bool, default=True)
    parser.add_argument('--wandb-entity', type=str)
    parser.add_argument('--wandb-project', type=str,
                        default="ldm")
    parser.add_argument('--log-freq', type=int, default=10)
    parser.add_argument('--save-freq', type=int, default=150)
    parser.add_argument('--debug', action='store_true')
    return parser

def parse_args(parser):
    args = parser.parse_args()

    if args.debug:
        args.workers = 0
        args.batch_size = 2
        args.epochs = 2
        args.log_freq = 1
        args.save_freq = 1
        args.on_wandb = False
        args.run_name = "debug"
    return vars(args)
