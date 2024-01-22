import torch
from utils.model import read_model_config
from models.ldm import LatentDiffusion
from models.vqgan import VQModel
from utils.model import load_weights
from models.modules.ddim import DDIMSampler
from utils.model import get_ae,  load_ae_checkpoint, read_model_config
from utils.image import *
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
import gradio as gr
import numpy as np
import torchvision.transforms.functional as TF
from einops import rearrange


device = 'cuda'
timesteps = 1000
ddim_steps = 100
ddim_eta = 1.0
ae_config = 'vqvae-f4.yaml'
ae_ckpt = 'weights/autoencoder/vqvae-f4-200.pt'

unet_config = 'unet-cldm-mask.yaml'
ldm_ckpt = 'weights/ldm/cldm-w-bg.pt'

ae_kwargs = read_model_config(ae_config)
ae_kwargs.update({"ae_ckpt": ae_ckpt})

unet_kwargs = read_model_config(unet_config)

default_kwargs = {
    "device": device,
}

ldm_kwargs = {
    "timesteps": timesteps,
    "unet_config": unet_kwargs,
    "ddim_steps": ddim_steps,
    "ddim_eta": ddim_eta,
    "device": device,
    "ldm_ckpt": ldm_ckpt,
}

ae = get_ae(ae_kwargs).to(device)
ae.eval()
load_ae_checkpoint(ae, ae_kwargs["ae_ckpt"], device)

# Instantiate LDM
ldm = LatentDiffusion(**ldm_kwargs).to(device)
num_timesteps = ldm.num_timesteps

checkpoint = torch.load(ldm_ckpt)
load_weights(ldm, checkpoint['ldm'])

ddim_steps = ldm_kwargs["ddim_steps"]
ddim_eta = ldm_kwargs["ddim_eta"]

ldm.eval()

is_unconditional = ldm_kwargs['unet_config']["is_unconditional"]
use_cross_attn = ldm_kwargs['unet_config']["use_cross_attn"]

ddim = DDIMSampler(ldm)

def convert_mask(mask):
    mask = torch.tensor(mask).float() / 255.
    # mask /= 255.
    mask = rearrange(mask, 'h w c -> c h w')
    mask = TF.resize(mask, (128, 128))
    r,g,b = mask
    r *= 1
    g *= 2
    b *= 3
    mask, _ = torch.max(torch.stack([r,g,b]), dim=0, keepdim=True)
    mask = mask.unsqueeze(0)
    return mask.to(device)

@torch.no_grad()
def encode_to_z(ae, x):
    z = ae.encode(x)
    if isinstance(ae, VQModel):
        z = z[0]
    return z

@torch.no_grad()
def encode_cond(mask, image, is_unconditional=False, use_cross_attn=True, ignore_global=False):  # TODO Move to LDM
    if is_unconditional:
        return {"global": None, "local": None}
    cond = {}
    zmask = encode_to_z(ae, mask)
    cond['local'] = zmask

    if image is None:
        image = torch.zeros_like(mask)
    cond['global'] = ldm.img_encoder(image)
    if ignore_global:
        cond['global'] = torch.zeros_like(cond['global'])
    #     cond['global'] = None
    return cond

def generate(mask=None, img=None, num_samples=2, ddim_steps=100, seed=42, ddim_eta=1.0, bs=1):
    ignore_global = False
    is_unconditional = False

    if mask is not None:
        mask = convert_mask(mask)
    if mask is None:
        is_unconditional = True
    if img is None:
        ignore_global = True

    c = encode_cond(mask, img, is_unconditional, ignore_global=ignore_global)
    shape = [ldm.unet.out_channels,
                ldm.unet.image_size,
                ldm.unet.image_size]

    with torch.no_grad():
        gen_latents, _ = ddim.sample(
            ddim_steps, bs, shape, c, eta=ddim_eta)
    generated = ae.decode(gen_latents)
    generated = [TF.to_pil_image(g.cpu().clamp(0,1)) for g in generated]
    return generated

def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## RADiff conditional inference")
            gr.Markdown("""Upload the mask in PNG with the following color coding:
                        - Blue: Extended source 
                        - Green: Point source 
                        - Red: Spurious source""")
    with gr.Row():
        with gr.Column():
            # canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
            # canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
            # create_button = gr.Button(label="Start", value='Open drawing canvas!')
            # input_sketch = gr.Image(source='upload', type='numpy', tool='color-sketch', label="Mask")

            input_sketch = gr.Image(source='upload', type='numpy', label="Mask")
            # gr.Markdown(value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
            #                   'Just click on the small pencil icon in the upper right corner of the above block.')
            # create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_sketch])
            input_image = gr.Image(source='upload', type='numpy', label="Background")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_sketch, input_image, num_samples, ddim_steps, seed, eta]
    run_button.click(fn=generate, inputs=ips, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')
