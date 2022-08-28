# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


import torch
import torch.nn.functional as F
from diffusers.pipeline_utils import DiffusionPipeline
from einops import rearrange
from tqdm.auto import tqdm


class PatchedDDIMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, kernel_size=64, stride=16, batch_size=128):
        super().__init__()

        scheduler = scheduler.set_format("pt")
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        shape,
        cond=None,
        generator=None,
        torch_device=None,
        eta=0.0,
        num_inference_steps=50,
        output_type=None,
    ):
        B, C, H, W = shape

        # eta corresponds to η in paper and should be between [0, 1]
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop, if has cond, concatenate
        # them in channel dim. image.shape = B C H W, C = 6 if concatenated
        image = torch.randn(shape, generator=generator).to(torch_device)
        if cond is not None:
            cond = cond.to(torch_device)

        # create mask for overlaped patches
        mask = torch.ones((1, 1, H, W)).to(torch_device)
        mask = mask.unfold(2, self.kernel_size, self.stride
                           ).unfold(3, self.kernel_size, self.stride)
        mask = rearrange(mask, 'b c nh nw h w -> b (c h w) (nh nw)')
        mask = F.fold(mask, output_size=(H, W), kernel_size=self.kernel_size,
                      stride=self.stride)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # 1. create overlapped patches from images
            # https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261/2

            if cond is not None:
                image = torch.cat([image, cond], dim=1)
            patches = image.unfold(2, self.kernel_size, self.stride
                                   ).unfold(3, self.kernel_size, self.stride)
            patches = rearrange(patches, 'b c nh nw h w -> (b nh nw) c h w')
            patch_list = torch.split(patches, self.batch_size, dim=0)
            
            res = []
            for patch in patch_list:
                # 2. predict noise from patches
                # NOTE: suppose img and cond have the same shape
                img, c = torch.chunk(patch, 2, dim=1)
                model_output = self.predict(img, c, t)
                # 3. get previous patches' mean and add variance depending on eta
                patch = self.scheduler.step(model_output, t, img, eta)["prev_sample"]
                res.append(patch)

            patches = torch.vstack(res)

            # 4. restore the whole image
            image = rearrange(patches, '(b k) c h w -> b (c h w) k', b=B)
            image = F.fold(image, (H, W), self.kernel_size, stride=self.stride)
            image /= mask

        image = (image / 2 + 0.5).clamp(0, 1).cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == 'numpy':
            image = image.numpy()

        return {"sample": image}

    def predict(self, img, cond, t):
        if cond is not None:
            input = torch.cat([img, cond], dim=1)
        return self.unet(input, t)['sample']
