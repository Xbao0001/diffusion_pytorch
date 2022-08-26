import builtins
import math
import os

import accelerate
import hydra
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from diffusion.models.ddpm_model import Model
from diffusion.pipelines import DDPMPipeline
from diffusion.schedulers import DDPMScheduler


@hydra.main(version_base=None, config_path='configs', config_name='cifar10.yaml')
def init_and_run(cfg):
    # save everything for this exp, except hydra
    cfg.exp_dir = os.path.join(cfg.exp_dir, cfg.name)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    cfg.save_dir = os.path.join(cfg.exp_dir, cfg.save_dir)  # save ckpts
    os.makedirs(cfg.save_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        log_with="wandb" if cfg.wandb else "tensorboard",
        logging_dir=cfg.exp_dir,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )
    accelerate.utils.set_seed(cfg.seed, device_specific=True)

    if not accelerator.is_main_process:
        def print_pass(*args, **kwargs): pass
        builtins.print = print_pass

    if accelerator.is_main_process:
        if cfg.wandb:
            config = OmegaConf.to_container(cfg, resolve=True,
                                            throw_on_missing=True)
            init_kwargs = {'wandb': {'name': cfg.name, 'dir': cfg.exp_dir}}
            accelerator.init_trackers(cfg.project, config=config,
                                      init_kwargs=init_kwargs)
        else:  # tensorboard
            accelerator.init_trackers('tensorboard')

    main(cfg, accelerator)


def main(cfg, accelerator: Accelerator):
    augmentations = TF.Compose([TF.RandomHorizontalFlip(), 
                                TF.ToTensor(), 
                                TF.Normalize([0.5], [0.5])])

    if cfg.data.name is not None:
        dataset = load_dataset(cfg.data.name, cache_dir=cfg.data.cache_dir, 
                               split="train")
    else:
        raise NotImplementedError()

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = DataLoader(dataset, batch_size=cfg.data.train_batch_size, 
                                  shuffle=True)
    
    model = Model(**cfg.model)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, 
                                 weight_decay=cfg.optim.weight_decay, 
                                 eps=cfg.optim.eps,)

    lr_scheduler = get_scheduler(
        cfg.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.optim.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.train.num_epochs
                            ) // cfg.train.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    ema_model = EMAModel(model, inv_gamma=cfg.ema.ema_inv_gamma, 
                         power=cfg.ema.ema_power, 
                         max_value=cfg.ema.ema_max_decay)

    global_step = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), 
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for _, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if cfg.train.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
                "step": global_step
            }
            if cfg.train.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.train.save_images_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model 
                                                  if cfg.train.use_ema else model),
                    scheduler=noise_scheduler,
                )

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(generator=generator, shape=(36, 3, 32, 32))["sample"]

                # denormalize the images and save to tensorboard
                if cfg.wandb: 
                    accelerator.log({'test_samples': wandb.Image(make_grid(images, 6))})
                else:
                    images_processed = (images.numpy() * 255).round().astype("uint8")
                    accelerator.trackers[0].writer.add_images(
                        "test_samples", images_processed, epoch
                    )

            if epoch % cfg.train.save_model_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                 accelerator.save(pipeline.unet.state_dict(), f'{cfg.save_dir}/epoch_{epoch:06d}.pth')
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    init_and_run()
