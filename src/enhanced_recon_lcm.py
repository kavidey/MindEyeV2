# %%
import os
os.environ["HF_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams["savefig.bbox"] = "tight"
from tqdm.auto import tqdm

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.utils
from accelerate import Accelerator, DeepSpeedPlugin


from diffusers import AutoPipelineForText2Image

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
enhancer_name = "lcm_dreamshaper"
base_filename = f"{enhancer_name}/{model_name}_all_enhancedrecons.{enhancer_name}"
all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")
all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
# %% Setup diffusion pipeline
pipe = AutoPipelineForText2Image.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(
    device
)
pipe.set_progress_bar_config(disable=True)
# %% Diffusion parameters
steps = [1,2,4,8,20]
for i in range(len(steps)):
    num_inference_steps = steps[i]
    filename = f"{base_filename}.{num_inference_steps}-steps"
    steps
    strength = 0.5
    guidance_scale = 7.5
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

            init_latents = image_to_latents(image, pipe)
            # Add noise to latents based on `starting_noise_level`
            noise = torch.randn(init_latents.shape).to(device)
            latent_timestep = torch.tensor(starting_timestep).to(device)
            latents = pipe.scheduler.add_noise(init_latents, noise, latent_timestep)

            enhanced_image = pipe(
                caption,
                latents=latents,
                strength=strength,
                num_inference_steps=4,
                original_inference_steps=original_inference_steps,
                timesteps=torch.linspace(
                    starting_timestep, 0, num_inference_steps + 1, dtype=int
                )[:-1],
                output_type="pt",
            ).images[0]
            enhanced_image = torch.tensor(enhanced_image).cpu()[None]

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
                ax1.imshow(transforms.ToPILImage()(all_recons[i]))
                ax1.axis("off")

                ax2 = grid[i * 2 + 1]
                ax2.imshow(transforms.ToPILImage()(all_enhancedrecons[i]))
                ax2.axis("off")

            plt.suptitle("Original (left) and Enhanced (Right)", fontsize=30, y=0.95)
            plt.show()
    # %%
    all_enhancedrecons = transforms.Resize((256, 256))(all_enhancedrecons).float()
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
            "original_inference_steps": original_inference_steps,
            "starting_noise_level": starting_noise_level
        },
        f"evals/{model_name}/{filename}.settings.pt",
    )
    print(f"saved evals/{model_name}/{filename}.pt")
# %%
