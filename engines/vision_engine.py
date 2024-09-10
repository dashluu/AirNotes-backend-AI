import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from engines.stable_diffusion import StableDiffusionXL


class VisionEngine:
    def save_img(self, tensor, img_name) -> Image:
        img = Image.fromarray(tensor)
        img.save(img_name)
        return img

    def __text_to_img(self, text):
        # Stable diffusion XL
        sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=False)

        # Quantization
        nn.quantize(
            sd.text_encoder_1, class_predicate=lambda _, m: isinstance(m, nn.Linear)
        )
        nn.quantize(
            sd.text_encoder_2, class_predicate=lambda _, m: isinstance(m, nn.Linear)
        )
        nn.quantize(sd.unet, group_size=32, bits=8)

        cfg = 0.0
        steps = 2

        # Ensure that models are read in memory if needed
        sd.ensure_models_are_loaded()

        # Generate the latent vectors using diffusion
        latents = sd.generate_latents(
            text,
            n_images=1,
            cfg_weight=cfg,
            num_steps=steps,
            seed=None,
            negative_text="",
        )

        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)

        # help in memory constrained systems by reusing the memory kept by the unet and the text encoders.
        del sd.text_encoder_1
        del sd.text_encoder_2
        del sd.unet
        del sd.sampler
        # Memory used by UNet in GBs
        peak_mem_unet = mx.metal.get_peak_memory() / 1024**3

        # Decode them into images
        decoded = []
        for i in tqdm(range(0, 1, 1)):
            decoded.append(sd.decode(x_t[i : i + 1]))
            mx.eval(decoded[-1])
        peak_mem_overall = mx.metal.get_peak_memory() / 1024**3

        # Arrange them on a grid
        x = mx.concatenate(decoded, axis=0)
        x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
        B, H, W, C = x.shape
        x = x.reshape(1, B, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(H, B * W, C)
        x = (x * 255).astype(mx.uint8)
        # Report the peak memory used during generation
        print(f"Peak memory used for the unet: {peak_mem_unet:.3f}GB")
        print(f"Peak memory used overall:      {peak_mem_overall:.3f}GB")
        return np.array(x)

    def text_to_img(self, text, img_name):
        # Generate image tensor from text
        t = self.__text_to_img(text)
        # Save them to disc
        self.save_img(t, img_name)
