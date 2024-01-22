<div align="center">    
 
# RADiff: Controllable Diffusion Models for Radio Astronomical Map Generation    
[Renato Sortino](https://rensortino.github.io),  [Thomas Cecconello](http://www.perceivelab.com/profile/tcecconello), [Andrea DeMarco](https://www.um.edu.mt/profile/andreademarco), [Giuseppe Fiameni](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=Se2mLvIAAAAJ), [Andrea Pilzer](https://andrea-pilzer.github.io/about/), [Andrew M. Hopkins](https://researchers.mq.edu.au/en/persons/andrew-hopkins), [Daniel Magro](https://github.com/DanielMagro97), [Simone Riggi](https://www.oact.inaf.it/simone-riggi/), [Eva Sciacca](https://www.oact.inaf.it/eva-sciacca/), Adriano Ingallinera, [Cristobal Bordiu](https://scholar.google.es/citations?user=W18yO88AAAAJ&hl=es), [Filomena Bufano](https://www.researchgate.net/profile/Filomena-Bufano), [Concetto Spampinato](https://www.dieei.unict.it/docenti/concetto.spampinato?eng)

[![Paper](http://img.shields.io/badge/paper-arxiv.2307.02392-B31B1B.svg)](https://arxiv.org/abs/2307.02392)
<!-- [![Conference](http://img.shields.io/badge/{CONFERENCE_NAME}-4b44ce.svg)](https://{CONFERENCE_PROCEEDING_LINK}) -->

<!--  
Conference   
-->   
</div>

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
 
## Overview   
This repository contains the official PyTorch implementation of the paper titled __"RADiff: Controllable Diffusion Models for Radio Astronomical Maps Generation"__. In the paper, we propose an approach based on diffusion models to augment small datasets for training segmentation methods in radio astronomy.

<br/>

![Teaser image](docs/teaser.gif)

## Requirements
1. Clone the repository:
 ```bash
 git clone https://github.com/SKA-INAF/radiff.git
 cd radiff
 ```

2. Create a conda environment and install the required dependencies:
```bash
conda env create -f environment.yaml
conda activate radiff
```


## Dataset

The data used to run the experiments for this paper is under privacy constraints and we are not allowed to publish it. 
However, the model can be trained on any collection of radio astronomical images in FITS format if it presents the folder structure described in this section.

Data should be put in the `data` folder. This folder should contain all the images and annotations, and a text file where each line defines the path of the images, one for train (`train.txt`) and one for validation (`val.txt`). The structure of the folder should be as follows:

```bash
data
├── train.txt
├── val.txt
├── images
│   ├── img0001.fits
│   ├── img0002.fits
│   ├── img0003.fits
│   └── ...
└── annotations
    ├── mask_img0001.json
    ├── mask_img0001_obj1.fits
    ├── mask_img0002.json
    ├── mask_img0002_obj1.fits
    ├── mask_img0002_obj2.fits
    ├── mask_img0002_obj3.fits
    ├── mask_img0003.json
    ├── mask_img0003_obj1.fits
    ├── mask_img0003_obj2.fits
    ├── ...
    └──
```

The `images` folder contains the 128x128 images to be used for training the model while the `annotations` folder contains information about each image in JSON format (class, bbox coordinates, flux intensity). Additionally, each FITS file contains the segmentation mask of each object.
Note that this folder structure is adapted to the DataLoader in this implementation but this can be adapted to another file structure.



## Pretrained Models
| Model        | {METRIC1}           | {METRIC2}                      | Link                                                                                                                                                                                   | Comments                                        
|--------------|---------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| Autoencoder (VQ-VAE f4) | 5.11           | 3.29                          |    [link](https://drive.google.com/file/d/18Y9KpUSk7hoR1BxjDsEaphBbcNWSOyUU/view?usp=drive_link)     |                                                 |  


## Usage

### Inference

The implementation supports two inference modes: 
1. CLI inference, allowing to iterate over a folder of masks and generate an image for each one
2. Interactive inference, more user-friendly but supporting only one mask at a time

#### CLI inference

TODO

#### Interactive Interface

Run the interface using [Gradio](https://www.gradio.app/)
```bash
python gradio_app.py
```

### Train on custom datasets

#### Train the Autoencoder
The first step is to train the autoencoder to reconstruct the images and prepare the latent space for the diffusion model. To do so, run the following command and specify the configuration file for its architecture:

```bash
python scripts/train_ae.py \ 
--ae-config vae-f4.yaml \
--dataset train.txt \
--run-dir {OUTPUT_DIR} \ 
--run-name {RUN_NAME} \
--on_wandb
```

### Train the diffusion model

Once the autoencoder is capable of reconstructing the images, we can train the diffusion model. To do so, run the following command by specifying, as done earlier, the configuration file:

```bash
python scripts/train_ldm.py \ 
--ae-config vae-f4.yaml \
--ae-ckpt weights/autoencoder/vae-f4.pt \
--unet-config unet-cldm-mask.yaml \
--dataset train.txt \
--run-dir {OUTPUT_DIR} \ 
--run-name {RUN_NAME} \
--on_wandb
```

### Additional Experiments

We provide additional experiments to test the performance of the model on different tasks. To run them, use the following commands:

#### Train DDPM for unconditional image generation

```bash
python scripts/train_ddpm.py \ 
--unet-config unet-pixel.yaml \
--dataset train.txt \
--run-dir {OUTPUT_DIR} \ 
--run-name {RUN_NAME} \
--on_wandb
```

#### Train DDPM for mask generation

```bash
python scripts/train_ddpm.py \ 
--unet-config unet-pixel-masks.yaml \
--dataset train.txt \
--run-dir {OUTPUT_DIR} \ 
--run-name {RUN_NAME} \
--on_wandb
```

#### Train LDM for unconditional image generation

```bash
python scripts/train_ldm.py \ 
--ae-config vae-f4.yaml \
--ae-ckpt weights/autoencoder/vae-f4.pt \
--unet-config unet-ldm.yaml \
--dataset train.txt \
--run-dir {OUTPUT_DIR} \ 
--run-name {RUN_NAME} \
--on_wandb
```

## Results

TODO

## BibTeX

```
@article{sortino2023radiff,
  title={RADiff: Controllable Diffusion Models for Radio Astronomical Maps Generation},
  author={Sortino, Renato and Cecconello, Thomas and DeMarco, Andrea and Fiameni, Giuseppe and Pilzer, Andrea and Hopkins, Andrew M and Magro, Daniel and Riggi, Simone and Sciacca, Eva and Ingallinera, Adriano and others},
  journal={arXiv preprint arXiv:2307.02392},
  year={2023}
}
```
