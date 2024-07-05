# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams["savefig.bbox"] = "tight"
from tqdm.auto import tqdm

import torch
import torchvision.transforms as T
from accelerate import Accelerator, DeepSpeedPlugin

from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image
from diffusers.utils import load_image

# This relies in the TCD Scheduler available here: https://github.com/jabir-zheng/TCD?tab=readme-ov-file
from scheduling_tcd import TCDScheduler

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)


# %% Helper Functions
def image_to_latents(image, pipe):
    # Preprocess image for VAE
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=device)

    # Get VAE latents
    init_latents = pipe.vae.encode(image)
    init_latents = init_latents.latent_dist.sample()
    init_latents = pipe.vae.config.scaling_factor * init_latents
    init_latents = torch.cat([init_latents], dim=0)

    return init_latents


# %% Load image-caption pairs
plot = False

model_name = "final_subj01_pretrained_40sess_24bs"
enhancer_name = "tcd_sdxlv1"
base_filename = f"{enhancer_name}/{model_name}_all_enhancedrecons.{enhancer_name}"
all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")
all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
# %% Setup diffusion pipeline
device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"
# %%
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id).to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe).to("cuda")

pipe.set_progress_bar_config(disable=True)
img2img_pipe.set_progress_bar_config(disable=True)
# %% Diffusion parameters
# steps = [1,2,4,8,20]
steps = [4,8,20]
for i in range(len(steps)):
    num_inference_steps = steps[i]
    filename = f"{base_filename}.{num_inference_steps}-steps"
    strength = 0.8
    eta = 0.3
    guidance_scale = 0
    original_inference_steps = 50
    starting_noise_level = 0.25  # 0.5 matches Mind-Eye2 paper
    starting_timestep = int(1000 * starting_noise_level)
    # %%
    all_enhancedrecons = None
    with torch.no_grad():
        for img_idx in tqdm(range(len(all_recons))):
            image = all_recons[img_idx]
            caption = all_predcaptions[img_idx]

            # plt.imshow(torch.swapaxes(image.T,0,1))

            image_rescaled = T.Resize(1024)(image)

            enhanced_image = img2img_pipe(
                prompt=[caption],
                image=image_rescaled[None],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                # Eta (referred to as `gamma` in the paper) is used to control the stochasticity in every step.
                # A value of 0.3 often yields good results.
                # We recommend using a higher eta when increasing the number of inference steps.
                eta=eta, 
                generator=torch.Generator(device=device).manual_seed(0),
                output_type="pt",
            ).images[0]

            enhanced_image = T.Resize(256)(enhanced_image)
            enhanced_image = enhanced_image.cpu()[None]

            if all_enhancedrecons is None:
                all_enhancedrecons = enhanced_image
            else:
                all_enhancedrecons = torch.vstack((all_enhancedrecons, enhanced_image))
        # %%
        if plot:
            fig = plt.figure(figsize=(20, 10))
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(5, 10),
                axes_pad=0.1,
            )
            for i in range(len(grid) // 2):
                ax1 = grid[i * 2]
                ax1.imshow(T.ToPILImage()(all_recons[i]))
                ax1.axis("off")

                ax2 = grid[i * 2 + 1]
                ax2.imshow(T.ToPILImage()(all_enhancedrecons[i]))
                ax2.axis("off")

            plt.suptitle("Original (left) and Enhanced (Right)", fontsize=30, y=0.95)
            plt.show()
    # %%
    all_enhancedrecons = T.Resize((256, 256))(all_enhancedrecons).float()
    print("all_enhancedrecons", all_enhancedrecons.shape)
    torch.save(
        all_enhancedrecons,
        f"evals/{model_name}/{filename}.pt",
    )
    torch.save(
        {
            "pipe": str(pipe),
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "eta": eta,
            "original_inference_steps": original_inference_steps,
            "starting_noise_level": starting_noise_level
        },
        f"evals/{model_name}/{filename}.settings.pt",
    )
    print(f"saved evals/{model_name}/{filename}.pt")
# %%
