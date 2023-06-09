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
from diffusers.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm


class DDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self, 
        shape, 
        cond=None, 
        generator=None, 
        torch_device=None, 
        output_type=None,
        **kwargs,
    ):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)

        for k, v in kwargs.items(): 
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(shape, generator=generator).to(torch_device)
        if cond is not None: 
            cond = cond.to(torch_device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.predict(image, cond, t, **kwargs)

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, **kwargs)["prev_sample"]

        image = (image / 2 + 0.5).clamp(0, 1).cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == 'numpy':
            image = image.numpy()

        return {"sample": image}

    def predict(self, img, cond, t, **kwargs):
        if cond is not None:
            img = torch.cat([img, cond], dim=1)
        return self.unet(img, t, **kwargs)['sample']
        