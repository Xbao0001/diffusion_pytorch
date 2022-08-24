import random
from typing import Any, Tuple, Optional, Callable

import torch
from PIL import Image

from torchvision.datasets import CIFAR10


class CIFAR10(CIFAR10):
    def __init__(
        self, 
        root: str, 
        train: bool = True, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
        download: bool = False,
        num_timesteps: int = 1000
    ) -> None:
        self.num_timesteps = num_timesteps
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        timestep = random.randint(0, self.num_timesteps)
        noise = torch.randn_like(img)
        return img, noise, timestep, target
