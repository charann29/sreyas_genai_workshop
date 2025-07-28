import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import requests
from io import BytesIO

class ImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

    def generate_image(self, prompt, negative_prompt="", steps=50, guidance=7.5):
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512
        ).images[0]

        return image

    def generate_variations(self, prompt, num_images=4):
        images = self.pipe(
            prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=50
        ).images

        return images

    def style_transfer(self, prompt, style="oil painting"):
        styled_prompt = f"{prompt}, {style} style"
        return self.generate_image(styled_prompt)

# Usage
img_gen = ImageGenerator()
image = img_gen.generate_image("A serene mountain landscape at sunset")
image.save("generated_landscape.png")

# Generate variations
variations = img_gen.generate_variations("A futuristic city skyline")
for i, img in enumerate(variations):
    img.save(f"city_variation_{i}.png")
