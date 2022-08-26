import builtins
import os
import random

import accelerate
import hydra
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from diffusion.data import get_dataloader
from diffusion.models.ddpm_model import Model
from diffusion.pipelines import DDPMPipeline
from diffusion.schedulers import DDPMScheduler


@hydra.main(version_base=None, config_path='configs', config_name='all_weather.yaml')
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
    ############################################################################
    train_dataloader, val_data_dict = get_dataloader(**cfg.data)

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

    ############################################################################
    global_step = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for _, (img, gt, noise, t, _) in enumerate(train_dataloader):
            img = torch.flatten(img, end_dim=1)
            gt = torch.flatten(gt, end_dim=1)
            noise = torch.flatten(noise, end_dim=1)
            t = t.reshape(-1, 1).repeat(1, cfg.data.n).flatten()

            noisy_images = noise_scheduler.add_noise(gt, noise, t)

            with accelerator.accumulate(model):
                # Predict the noise residual
                input = torch.cat([noisy_images, img], dim=1)
                noise_pred = model(input, t)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr = lr_scheduler.get_last_lr()[0]
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                if cfg.train.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr,
                "step": global_step
            }
            if cfg.train.use_ema:
                logs["ema_decay"] = ema_model.decay
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()

        accelerator.wait_for_everyone()
    ############################################################################

        # Generate sample images for visual inspection
        # TODO: sample different val data using multiple gpus, log in main process
        if accelerator.is_main_process:
            if epoch % cfg.train.save_images_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                # TODO: use ddim
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model 
                                                  if cfg.train.use_ema else model),
                    scheduler=noise_scheduler,
                )
                
                raindrop_val_dataset, snow_val_dataset = val_data_dict.values()
                
                input_raindrop, gt_raindrop, *_ = raindrop_val_dataset[0]
                output_raindrop = pipeline(
                    shape=(cfg.data.n, 3, cfg.data.patch_size, cfg.data.patch_size),
                    cond=input_raindrop,
                    generator=torch.manual_seed(0),
                )["sample"]

                idx = random.randint(0, len(snow_val_dataset))
                input_snow, gt_snow, *_ = snow_val_dataset[idx]
                output_snow = pipeline(
                    shape=(cfg.data.n, 3, cfg.data.patch_size, cfg.data.patch_size),
                    cond=input_snow,
                )["sample"]


                # denormalize the images and save to tracker
                if cfg.wandb:
                    accelerator.log({
                        'de_raindrop': wandb.Image(make_grid(output_raindrop, 4)),
                        'gt_raindrop': wandb.Image(make_grid(gt_raindrop, 4)),
                        'raindrop': wandb.Image(make_grid(input_raindrop, 4)),
                        'de_snow': wandb.Image(make_grid(output_snow, 4)),
                        'gt_snow': wandb.Image(make_grid(gt_snow, 4)),
                        'snow': wandb.Image(make_grid(input_snow, 4)),
                    })
                else:
                    images_processed = (output_raindrop.numpy() * 255).round().astype("uint8")
                    accelerator.trackers[0].writer.add_images("de_raindrop", images_processed, epoch)
                    images_processed = (output_snow.numpy() * 255).round().astype("uint8")
                    accelerator.trackers[0].writer.add_images("de_snow", images_processed, epoch)

            if epoch % cfg.train.save_model_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                 accelerator.save(pipeline.unet.state_dict(), 
                           f'{cfg.save_dir}/epoch_{epoch:06d}.pth')
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    init_and_run()
