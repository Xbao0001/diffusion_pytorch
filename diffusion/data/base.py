import random
from typing import Callable, Optional

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        patch_size: int = 64, 
        n: int = 16, 
        transform: Optional[Callable] = None, 
        parse_patches: bool = True,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        self.root = root
        self.patch_size = patch_size
        self.n = n
        self.transform = transform
        self.parse_patches = parse_patches
        self.num_timesteps = num_timesteps

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_paths_and_name(self, idx):
        """return input_path, gt_path, img_name
        """
        raise NotImplementedError('plz implement this method w.r.t docstring')

    def __getitem__(self, idx):
        input_path, gt_path, img_name = self.get_paths_and_name(idx)        

        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.parse_patches: # for train
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_patches = self.n_random_crops(input_img, i, j, h, w)
            gt_patches = self.n_random_crops(gt_img, i, j, h, w)

            if self.transform is not None:
                input_patches = torch.vstack([self.transform(img).unsqueeze(0) for img in input_patches])
                gt_patches = torch.vstack([self.transform(img).unsqueeze(0) for img in gt_patches])
            else: 
                raise ValueError("plz specify transform")

            timestep = random.randint(0, self.num_timesteps)
            noise = torch.randn_like(gt_patches)
            
            return input_patches, gt_patches, noise, timestep, img_name
        else: # for infer, test one image at a time
            # resizing images to multiples of 16
            wd_new, ht_new = input_img.size
            # NOTE: why need this? GPU memory limitation?
            # if ht_new > wd_new and ht_new > 1024: 
            #     wd_new = int(wd_new * 1024.0 / ht_new)
            #     ht_new = 1024
            # elif ht_new <= wd_new and wd_new > 1024:
            #     ht_new = int(ht_new * 1024.0 / wd_new)
            #     wd_new = 1024
            wd_new = int(wd_new / 16.0) * 16
            ht_new = int(ht_new / 16.0) * 16
            input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)
            if self.transform is not None:
                input_img = self.transform(input_img)
                gt_img = self.transform(gt_img)
            
            noise = torch.randn_like(gt_img)
            return input_img, gt_img, noise, img_name

    def __len__(self):
        raise NotImplementedError('plz return dataset size')
