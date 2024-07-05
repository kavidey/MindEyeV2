# %% Imports
import os

os.environ["HF_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/"

import sys
import numpy as np
import h5py
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append("generative_models/")
from generative_models.sgm.models.diffusion import DiffusionEngine

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

from diffusers.models.vae import Decoder
from diffusers import AutoencoderKL
from transformers import AutoProcessor

from models import PriorNetwork, BrainDiffusionPrior
from modeling_git import GitForCausalLMClipEmb
import utils

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)


# %% Model Classes
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, out_features) for input_size in input_sizes]
        )

    def forward(self, x, subj_idx):
        out = torch.cat(
            [self.linears[subj_idx](x[:, seq]).unsqueeze(1) for seq in range(seq_len)],
            dim=1,
        )
        return out


class BrainNetwork(nn.Module):
    def __init__(
        self,
        h=4096,
        in_dim=15724,
        out_dim=768,
        seq_len=2,
        n_blocks=4,
        drop=0.15,
        clip_size=768,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size

        self.mixer_blocks1 = nn.ModuleList(
            [self.mixer_block1(h, drop) for _ in range(n_blocks)]
        )
        self.mixer_blocks2 = nn.ModuleList(
            [self.mixer_block2(seq_len, drop) for _ in range(n_blocks)]
        )

        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True)
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)

        # This code is for blurry_recon, it needs to be here even if we're not using it
        # because the trained model will try to load weights for it
        self.blin1 = nn.Linear(h * seq_len, 4 * 28 * 28, bias=True)
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )
        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len), self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )

    def forward(self, x):
        # make empty tensors
        c, b, t = torch.Tensor([0.0]), torch.Tensor([[0.0], [0.0]]), torch.Tensor([0.0])

        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)

            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)

        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        c = self.clip_proj(backbone)

        b = self.blin1(x)
        b = self.bdropout(b)
        b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
        b = self.bnorm(b)
        b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
        b_aux = b_aux.view(len(b_aux), 49, 512)
        b = (self.bupsampler(b), b_aux)

        return backbone, c, b


class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


# %% Define Constants
data_path = Path("../data")

# Which subject to train on
subj = 1
model_name = "final_subj01_pretrained_40sess_24bs"

# Model parameters
hidden_dim = 4096
seq_len = 1

minibatch_size = 1
num_samples_per_image = 1
assert num_samples_per_image == 1
# %% Load data
voxels = {}
# Load hdf5 data for betas
f = h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r")
betas = f["betas"][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f"subj0{subj}"] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

num_test = 3000
test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"


def my_split_by_node(urls):
    return urls


test_data = (
    wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
    .decode("torch")
    .rename(
        behav="behav.npy",
        past_behav="past_behav.npy",
        future_behav="future_behav.npy",
        olds_behav="olds_behav.npy",
    )
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
)
test_dl = torch.utils.data.DataLoader(
    test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True
)
print(f"Loaded test dl for subj{subj}!\n")

# Prep images but don't load them all to memory
f = h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r")
images = f["images"]

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f"subj0{subj}"][behav[:, 0, 5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:, 0, 5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))
# %%
clip_seq_dim = 256
clip_emb_dim = 1664

model = MindEyeModule()
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim, seq_len=seq_len)
model.backbone = BrainNetwork(
    h=hidden_dim,
    in_dim=hidden_dim,
    seq_len=seq_len,
    clip_size=clip_emb_dim,
    out_dim=clip_emb_dim * clip_seq_dim,
)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim // 52  # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
    dim=out_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=clip_seq_dim,
    learned_query_mode="pos_emb",
)

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)
# %%
# Load pretrained model ckpt
tag = "last"
outdir = (data_path / "train_logs" / model_name).resolve().absolute()
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir / f"{tag}.pth", map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except:  # probably ckpt is saved using deepspeed format
    import deepspeed

    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir=outdir, tag=tag
    )
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")
# %%
autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256,
)
ckpt = torch.load(data_path/'sd_image_var_autoenc.pth')
autoenc.load_state_dict(ckpt)
autoenc.eval()
autoenc.requires_grad_(False)
autoenc.to(device)
utils.count_params(autoenc)
# %%
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(
    device
)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

clip_convert = CLIPConverter()
state_dict = torch.load(data_path / "bigG_to_L_epoch8.pth", map_location="cpu")[
    "model_state_dict"
]
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(
    device
)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict
# %%

config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
sampler_config["params"]["num_steps"] = 38

diffusion_engine = DiffusionEngine(
    network_config=network_config,
    denoiser_config=denoiser_config,
    first_stage_config=first_stage_config,
    conditioner_config=conditioner_config,
    sampler_config=sampler_config,
    scale_factor=scale_factor,
    disable_first_stage_autocast=disable_first_stage_autocast,
)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)
# %%
ckpt_path = f"{data_path}/unclip6_epoch0_step110000.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
diffusion_engine.load_state_dict(ckpt["state_dict"])

batch = {
    "jpg": torch.randn(1, 3, 1, 1).to(
        device
    ),  # jpg doesnt get used, it's just a placeholder
    "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
    "crop_coords_top_left": torch.zeros(1, 2).to(device),
}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)
# %%
# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for batch in tqdm(range(0, len(np.unique(test_images_idx)), minibatch_size)):
    # for batch in tqdm(range(0,10,minibatch_size)):
        uniq_imgs = np.unique(test_images_idx)[batch : batch + minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx == uniq_img)[0]
            if len(locs) == 1:
                locs = locs.repeat(3)
            elif len(locs) == 2:
                locs = locs.repeat(2)[:3]
            assert len(locs) == 3
            if voxel is None:
                voxel = test_voxels[None, locs]  # 1, num_image_repetitions, num_voxels
            else:
                voxel = torch.vstack((voxel, test_voxels[None, locs]))
        voxel = voxel.to(device)

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:, [rep]], 0)  # 0th index of subj_list
            backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
            if rep == 0:
                clip_voxels = clip_voxels0
                backbone = backbone0
                blurry_image_enc = blurry_image_enc0[0]
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
                blurry_image_enc += blurry_image_enc0[0]
        clip_voxels /= 3
        backbone /= 3
        blurry_image_enc /= 3

        # Save retrieval submodule outputs
        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels.cpu()
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))

        # Feed voxels through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(
            backbone.shape,
            text_cond=dict(text_embed=backbone),
            cond_scale=1.0,
            timesteps=20,
        )

        pred_caption_emb = clip_convert(prior_out)
        generated_ids = clip_text_model.generate(
            pixel_values=pred_caption_emb, max_length=20
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        print(generated_caption)

        # Feed diffusion prior outputs through unCLIP
        for i in range(len(voxel)):
            samples = utils.unclip_recon(
                prior_out[[i]],
                diffusion_engine,
                vector_suffix,
                num_samples=num_samples_per_image,
            )
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))
        
        blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)
        for i in range(len(voxel)):
            im = torch.Tensor(blurred_image[i])
            if all_blurryrecons is None:
                all_blurryrecons = im[None].cpu()
            else:
                all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))

# resize outputs before saving
imsize = 256
all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()

# saving
torch.save(all_recons, f"evals/{model_name}/{model_name}_all_recons.pt")
torch.save(all_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
torch.save(all_predcaptions, f"evals/{model_name}/{model_name}_all_predcaptions.pt")
torch.save(all_clipvoxels, f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
print(f"saved {model_name} outputs!")
# %%
all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")
all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
# %%
fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(5, 5),
    axes_pad=0.4,
)
for i,ax in enumerate(grid):
    ax.imshow(transforms.ToPILImage()(all_recons[i]))
    ax.axis("off")
    ax.set_title(all_predcaptions[i], wrap=True, fontsize=8, loc="center")
plt.suptitle("Unrefined Reconstructions")
plt.show()
# %%
fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(5, 5),
    axes_pad=0.4,
)
for i,ax in enumerate(grid):
    ax.imshow(transforms.ToPILImage()(all_blurryrecons[i]))
    ax.axis("off")
    ax.set_title(all_predcaptions[i], wrap=True, fontsize=8, loc="center")
plt.suptitle("Blurry Reconstructions")
plt.show()
# %%
