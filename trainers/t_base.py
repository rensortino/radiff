from utils.logger import Logger


class BaseTrainer:
    def __init__(
        self,
        logger_kwargs,
        train_loader,
        test_loader,
        device="cuda",
        epochs=300,
        save_freq=100,
        log_freq=5,
        debug=False,
    ):
        r"""
        Base class for all trainers

        Args:
            logger_kwargs: (dict) contains logger arguments
            train_loader: (torch.data.DataLoader) training dataloader
            test_loader: (torch.data.DataLoader) test dataloader
            device: (str) device to run training on
            epochs: (int) number of epochs to train for
            save_freq: (int) frequency at which to save intermediate checkpoints
            log_freq: (int) frequency at which to log last checkpoints
            debug: (bool) whether to run in debug mode
        """
        self.device = device
        self.start_epoch = 0
        self.num_epochs = epochs
        self.run_name = logger_kwargs["run_name"]
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.debug = debug

        # best checkpoint
        self.best_ckpt = {"loss": float("inf")}

        # setup logger
        self.logger = Logger(**logger_kwargs)

        # setup dataloaders
        self.train_loader = train_loader
        self.test_loader = test_loader

    def init_checkpoint(self, resume):
        if resume:
            self.load_checkpoint(resume)

    def update_best_ckpt(self, ckpt):
        if ckpt["loss"] < self.best_ckpt["loss"]:
            self.best_ckpt = ckpt

    def train(self):
        for i in range(self.start_epoch, self.num_epochs):
            # Run one epoch of training
            losses_epoch = self.train_one_epoch()
            # Log training metrics
            for key in losses_epoch.keys():
                self.logger.log_metric(f"train/{key}", losses_epoch[key], i)

            losses_epoch = self.evaluate(epoch=i)
            # Log checkpoint
            ckpt = self._make_ckpt(i, losses_epoch["loss"])
            self.update_best_ckpt(ckpt)
            if i % self.log_freq == 0:
                self.logger.save_ckpt(ckpt, f"last.pt")
            if (i + 1) % self.save_freq == 0:
                self.logger.save_ckpt(ckpt, f"{self.run_name}-{i+1}.pt")
                self.logger.save_ckpt(self.best_ckpt, f"best.pt")

    def evaluate(self, epoch=0):
        # Run one epoch of evaluation
        losses_epoch, samples = self.eval_one_epoch()

        # Log evaluation metrics
        for key in losses_epoch.keys():
            self.logger.log_metric(f"eval/{key}", losses_epoch[key], epoch)

        # Log sample images
        for k in samples.keys():
            self.logger.log_images(samples[k], "eval", k, epoch)
        return losses_epoch
