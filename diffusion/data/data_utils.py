from typing import Dict, Tuple

import torchvision.transforms as tf
from torch.utils.data import DataLoader, Dataset

from .all_weather import AllWeatherDataset
from .raindrop import RaindropValDataset
from .snow100k import Snow100K


def get_all_weather_dataloader(
    train_root = '/nas/datasets/weather/all_weather', 
    raindrop_val_root = '/nas/datasets/weather/raindrop/test_a', 
    snow_val_root = '/nas/datasets/weather/snow100k',
    patch_size: int = 64, 
    n: int = 16, 
    num_timesteps: int = 1000, 
    batch_size: int = 4,
    num_workers: int = 4,
    parse_patches: bool = True,
) -> Tuple[DataLoader, Dict[str, Dataset]]:

    transform=tf.ToTensor()

    train_dataset = AllWeatherDataset(
        root=train_root,
        patch_size=patch_size,
        n=n,
        transform=transform,
        parse_patches=parse_patches,
        num_timesteps=num_timesteps,
    )

    raindrop_val_dataset = RaindropValDataset(
        root=raindrop_val_root,
        patch_size=patch_size,
        n=n,
        transform=transform,
        parse_patches=parse_patches,
        num_timesteps=num_timesteps,
    )
    snow_val_dataset = Snow100K( 
        root=snow_val_root,
        patch_size=patch_size,
        n=n,
        transform=transform,
        parse_patches=parse_patches,
        num_timesteps=num_timesteps,
        mode='L',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        # persistent_workers=True,
    )

    # raindrop_val_loader = DataLoader(
    #     raindrop_val_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    #     # persistent_workers=True,
    # )

    # snow_val_loader = DataLoader(
    #     snow_val_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    #     # persistent_workers=True,
    # )
    return train_loader, {'raindrop': raindrop_val_dataset, 
                          'snow': snow_val_dataset}


def get_snow100k(
    root = '/nas/datasets/weather/snow100k', 
    val_mode='L',
    patch_size: int = 64, 
    n: int = 16, 
    num_timesteps: int = 1000, 
    batch_size: int = 4,
    num_workers: int = 4,
    parse_patches: bool = True,
) -> Tuple[DataLoader, Dataset]:

    transform = tf.ToTensor()
    
    train_dataset = Snow100K(
        root=root,
        patch_size=patch_size,
        n=n,
        transform=transform,
        parse_patches=parse_patches,
        num_timesteps=num_timesteps,
        mode='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataset = Snow100K(
        root=root,
        patch_size=patch_size,
        n=n,
        transform=transform,
        parse_patches=parse_patches,
        num_timesteps=num_timesteps,
        mode=val_mode,
    )
    return train_loader, val_dataset
