from diffusers import StableDiffusionPipeline
import torch

img_gen_model = "runwayml/stable-diffusion-v1-5"
img_gen = StableDiffusionPipeline.from_pretrained(img_gen_model, torch_dtype=torch.float16).to("cuda")


def text_to_img(text, img_name):
    img = img_gen(text).images[0]
    img.save(img_name)
