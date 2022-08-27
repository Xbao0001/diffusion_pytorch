import glob
import os
from typing import Callable, Optional

from .base import BaseDataset


class Snow100K(BaseDataset):
    def __init__(
        self, 
        root: str = '/nas/datasets/weather/snow100k', 
        patch_size: int = 64, 
        n: int = 16, 
        transform: Optional[Callable] = None, 
        parse_patches: bool = True,
        num_timesteps: int = 1000,
        mode: str = 'train',
    ):
        assert mode in ['train', 'L', 'M', 'S']
        super().__init__(root, patch_size, n, transform, 
                         parse_patches, num_timesteps)
        rel_path = {'train': 'train', 'L': 'test/Snow100K-L', 
                    'M': 'test/Snow100K-M', 'S': 'test/Snow100K-S'}
        self.root = os.path.join(self.root, rel_path[mode])
        self.input_paths = sorted(glob.glob(f"{self.root}/synthetic/*"))
        print(f'Snow100k-{mode} dataset size: {len(self.input_paths)}.')
    
    def get_paths_and_name(self, idx):
        input_path = self.input_paths[idx]
        gt_path = input_path.replace('synthetic', 'gt')
        img_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        return input_path, gt_path, img_name

    def __len__(self):
        return len(self.input_paths)
