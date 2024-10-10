from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Generate an image
prompt = "A fantasy landscape with mountains and a river"
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_image.png")
