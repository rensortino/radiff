from datetime import datetime
from pathlib import Path
from torchsummary.torchsummary import summary
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import sys
import torch
import wandb
from logzero import logger as lz_logger
import logzero

def to_pil(image):
    image = image / 2 + 0.5
    image = torch.clamp(image, 0, 1)

    return TF.to_pil_image(image.squeeze())

class Logger:

    def __init__(self, output_dir, run_name, on_wandb=False, wandb_entity="", wandb_project=""):

        self.out_dir = Path(output_dir)
        self.run_name = Path(run_name)
        self.on_wandb = on_wandb
        self.wandb_project = wandb_project
        self._set_dirs()

        logzero.logfile(self.out_dir / Path('output.log'))
        if on_wandb:
            wandb.login()
            wandb.init(entity=wandb_entity,
                       project=wandb_project, name=run_name)

        self._log_command()

    def _set_dirs(self):
        current_date = datetime.now()
        formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'
        test_name = f'{formatted_date}'
        self.out_dir = self.out_dir / self.run_name / Path(test_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log_command(self):
        with open(self.out_dir / Path('command.txt'), 'w') as out:
            cmd_args = ' '.join(sys.argv)
            out.write(cmd_args)
            out.write('\n')

    def log_image(self, pil_img, phase, title, step):
        pil_img.save(self.out_dir / Path(f'{title}.png'))
        if self.on_wandb:
            wandb.log({f'{phase}/{title}': wandb.Image(pil_img), 'epoch': step})

    def log_images(self, images, phase, title, step, size=None):
        wnb_images = []
        for i, im in enumerate(images[:8]):
            size = size if size is not None else im.shape[-1]
            im = TF.resize(im, size)
            save_path = self.out_dir / Path(f'{step}_{i}_{title}.png')
            save_image(im, save_path)
            wnb_images.append(wandb.Image(str(save_path)))
        
        if self.on_wandb:
            wandb.log({f'{phase}/{title}': wnb_images, 'epoch': step})


    def log_metric(self, title, metric, step):
        lz_logger.info(f'Epoch [{step}] - {title}: {metric}')
        if self.on_wandb:
            wandb.log({title: metric, 'epoch': step})

    def log_summary(self, model, input_size=(4,), tgt_size=(4, 260), batch_size=32):
        # Give no batch size in tgt_size
        summary(model, self.out_dir / Path('summary.txt'),
                input_size, tgt_size, batch_size)

    def log_lr(self, opt, epoch, title='train/lr'):
        lr = get_lr(opt)
        if self.on_wandb:
            wandb.log({title: lr, "epoch": epoch})

    def warning(self, text):
        lz_logger.warning(text)

    def info(self, text):
        lz_logger.info(text)

    def save_ckpt(self, ckpt, path='checkpoint.pt'):
        path = Path(path)
        lz_logger.info(f'Saving {path} in {self.out_dir}')
        torch.save(ckpt, self.out_dir / path)
        if self.on_wandb:
            wandb.save(str(self.out_dir / path))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    logger = Logger('logs_test', 'Test logging')
    logger.log_metric('Test metric', 0.5, 1)
    logger.info('This is an info message')
    logger.warning('This is a warning')
    ckpt = torch.randn(64, 3, 256, 256)
    logger.save_ckpt(ckpt, 'test.pt')
