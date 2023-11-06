import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from datasets.radiogalaxy import RGDataset
from datasets.utils import *
from utils.image import save_fits

warnings.simplefilter("ignore", category=VerifyWarning)
from utils.image import mask_to_rgb


class MaskOnlyDataset(Dataset):
    def __init__(self, data_dir, mask_paths, device):
        super().__init__()
        data_dir = Path(data_dir)
        self.device = device
        with open(mask_paths) as f:
            self.mask_paths = f.read().splitlines()
        self.mask_paths = [data_dir / p for p in self.mask_paths]

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        mask = fits.getdata(mask_path).astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        mask = TF.resize(mask, (128, 128))
        assert (
            mask.unique().max() <= 3
        ), "Mask must have one of the following values: 0, 1, 2, or 3"

        return mask


class MaskRGDataset(RGDataset):
    def __init__(self, data_dir, img_paths, img_size=128):
        super().__init__(data_dir, img_paths, img_size)
        self.img_size = img_size

        self.mask_transforms = T.Compose(
            [
                FromNumpy(),
                Unsqueeze(),
                T.Resize(
                    (img_size, img_size), interpolation=T.InterpolationMode.NEAREST
                ),
            ]
        )

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        img = fits.getdata(image_path)
        img = self.transforms(img)

        # Random crop
        # crop_size = self.img_size // 2
        h_crop_size = random.sample(range(self.img_size // 4, self.img_size), 1)[0]
        w_crop_size = random.sample(range(self.img_size // 4, self.img_size), 1)[0]
        t, l, h, w = T.RandomCrop.get_params(
            img, output_size=(h_crop_size, w_crop_size)
        )
        crop = TF.crop(img, t, l, h, w)
        r = self.img_size - w - l
        b = self.img_size - h - t
        masked_image = TF.pad(crop, padding=[l, t, r, b], fill=0)

        ann_path = str(image_path).replace("imgs", "masks").replace(".fits", ".json")
        ann_dir = Path(ann_path).parent
        ann_path = ann_dir / f'mask_{ann_path.split("/")[-1]}'
        with open(ann_path) as j:
            mask_info = json.load(j)

        bboxes = []
        categories = []
        masks = []

        W, H = mask_info["nx"], mask_info["ny"]

        for obj in mask_info["objs"]:
            bbox = [
                obj["bbox_x"] / W,
                obj["bbox_y"] / H,
                obj["bbox_w"] / W,
                obj["bbox_h"] / H,
            ]

            category = obj["class"]
            seg_path = ann_dir / obj["mask"]

            bboxes.append(bbox)
            categories.append(category)
            mask = fits.getdata(seg_path)

            mask = self.mask_transforms(mask.astype(np.float32))
            masks.append(mask)

        if "bkg" in str(image_path):
            mask = torch.zeros_like(img)
            masks.append(mask)

        mask, _ = torch.max(torch.stack(masks), dim=0)

        return img, mask, masked_image


if __name__ == "__main__":
    import sys

    sys.path.append("/home/rsortino/inaf/radio-diffusion")
    rgtrain = MaskRGDataset("data/rg-dataset/data", "data/rg-dataset/val_mask.txt")
    batch = next(iter(rgtrain))
    image, mask, masked_image = batch
    to_pil_image(image).save("image.png")
    rgb_mask = mask_to_rgb(mask)[0]
    to_pil_image(rgb_mask).save("mask.png")
    to_pil_image(masked_image[0]).save("masked.png")

    bs = 256

    loader = torch.utils.data.DataLoader(
        rgtrain, batch_size=bs, shuffle=False, num_workers=16
    )
    for i, batch in enumerate(loader):
        image, mask, masked_image = batch
        save_fits(image, f"tmp/{i}.fits")
        rgb_mask = mask_to_rgb(mask)
        nrow = int(math.sqrt(bs))