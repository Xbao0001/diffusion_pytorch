import torch
from dir.models.ddpm_model import *

from diffusers import DDPMPipeline, DDPMScheduler
from omegaconf import OmegaConf
from tqdm import trange


cfg = OmegaConf.load('./configs/cifar10.yaml')
model = Model(**cfg.model)
ckpt = torch.load('./pretrained/ddpm/cifar10_ema_model-790000.ckpt')
model.load_state_dict(ckpt)
model.sample_size = 32


noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
pipeline = DDPMPipeline(
    unet=model,
    scheduler=noise_scheduler,
)

batch_size = 100
num_iter = 125

for iter in trange(num_iter):
    images = pipeline(batch_size, output_type='pil')['sample']
    for i, img in enumerate(images):
        img.save(f'./results/cifar10/img3_{iter * batch_size + i:05d}.jpg')
