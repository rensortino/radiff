import torch
import torchvision.transforms.functional as TF
from astropy.io import fits
from PIL import Image

COLORS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def save_fits(image, path, invert_y=True):
    image = image.to("cpu").detach().numpy()
    if invert_y:
        image = image[:, ::-1, :]  # FITS reads the y axis in the opposite direction
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(f"{path}", overwrite=True)


def rgb_to_tensor(mask_path):
    mask = Image.open(mask_path)
    mask = TF.to_tensor(mask)
    r, g, b = mask
    r *= 1
    g *= 2
    b *= 3
    mask, _ = torch.max(torch.stack([r, g, b]), dim=0, keepdim=True)
    return mask


def mask_to_rgb(mask):
    rgb_mask = torch.zeros_like(mask, device=mask.device).repeat(1, 3, 1, 1)
    for i, c in enumerate(COLORS):
        color_mask = torch.tensor(c, device=mask.device).unsqueeze(1).unsqueeze(2) * (
            mask == i
        )
        rgb_mask += color_mask
    return rgb_mask
