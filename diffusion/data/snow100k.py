import glob
import os
from typing import Callable, Optional

from .base import BaseDataset


class SnowValDataset(BaseDataset):
    def __init__(
        self, 
        root: str = '/nas/datasets/weather/snow100k/test/Snow100K-L', 
        patch_size: int = 64, 
        n: int = 16, 
        transform: Optional[Callable] = None, 
        parse_patches: bool = True,
        num_timesteps: int = 1000,
    ):
        super().__init__(root, patch_size, n, transform, 
                         parse_patches, num_timesteps)

        self.input_paths = sorted(glob.glob(f"{self.root}/synthetic/*"))
        print(f'Snow100k val dataset size: {len(self.input_paths)}.')
    
    def get_paths_and_name(self, idx):
        input_path = self.input_paths[idx]
        gt_path = input_path.replace('synthetic', 'gt')
        img_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        return input_path, gt_path, img_name

    def __len__(self):
        return len(self.input_paths)
