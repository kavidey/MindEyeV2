# %%
import os
os.environ["HF_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/huggingface"
os.environ["TORCH_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/torch"

from pathlib import Path
import itertools
import textwrap
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from torchvision.models.feature_extraction import create_feature_extractor
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

### Multi-GPU config ###
local_rank = os.getenv("RANK")
if local_rank is None:
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)

accelerator = Accelerator(
    split_batches=False, mixed_precision="fp16"
)  # ['no', 'fp8', 'fp16', 'bf16']

print("PID of this process =", os.getpid())
device = accelerator.device
print("device:", device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == "NO"
num_devices = torch.cuda.device_count()
if num_devices == 0 or not distributed:
    num_devices = 1
num_workers = num_devices
print(accelerator.state)

print(
    "distributed =",
    distributed,
    "num_devices =",
    num_devices,
    "local rank =",
    local_rank,
    "world size =",
    world_size,
)
print = accelerator.print  # only print if local_rank=0


# %%
def wrap_title(title, wrap_width):
    return "\n".join(textwrap.wrap(title, wrap_width))


@torch.no_grad()
def two_way_identification(
    all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True
):
    preds = model(
        torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device)
    )
    reals = model(
        torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device)
    )
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    else:
        return success_cnt, len(all_images) - 1


# %%
plot = False

model_name = "final_subj01_pretrained_40sess_24bs"

enhancer_names = ["lcm_dreamshaper", "tcd_sdxlv1"]
steps = [1, 2, 4, 8, 20]

eval_path = Path("evals")

# ground truths
all_images = torch.load(eval_path / "all_images.pt")
all_captions = torch.load(eval_path / "all_captions.pt")
# %% Image <-> Brain retrieval evaluation
# clip_seq_dim = 256
# clip_emb_dim = 1664
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
# %% AlexNet
from torchvision.models import alexnet, AlexNet_Weights

alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(
    alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
)
alex_model.eval().requires_grad_(False)
# %% InceptionV3
from torchvision.models import inception_v3, Inception_V3_Weights

weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(
    inception_v3(weights=weights), return_nodes=["avgpool"]
)
inception_model.eval().requires_grad_(False)
# %% CLIP
import clip

clip_model, preprocess = clip.load("ViT-L/14")
# %% EfficientNet
import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(
    efficientnet_b1(weights=weights), return_nodes=["avgpool"]
)
eff_model.eval().requires_grad_(False)
# %% SwAV
swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"])
swav_model.eval().requires_grad_(False)
# %%
tests = itertools.product(enhancer_names, steps)
results = []
column_names = [
    "model_name",
    "enhancer_name",
    "num_inference_steps",
    "retrieval_fwd",
    "retrieval_fwd_ci",
    "retrieval_bwd",
    "retrieval_bwd_ci",
    "pixcorr",
    "ssim",
    "alexnet2",
    "alexnet5",
    "inceptionv3",
    "clip",
    "efficientnet",
    "swav"
]
for enhancer_name, num_inference_steps in tqdm(tests, total=(len(enhancer_names)*len(steps))):
    print(f"Testing: {enhancer_name}, {num_inference_steps} steps")
    result = [model_name, enhancer_name, num_inference_steps]
    all_recons = torch.load(
        eval_path
        / model_name
        / enhancer_name
        / f"{model_name}_all_enhancedrecons.{enhancer_name}.{num_inference_steps}-steps.pt"
    )
    # unenhanced recons
    # all_recons = torch.load(eval_path / model_name / f"{model_name}_all_recons.pt")
    all_blurryrecons = torch.load(
        eval_path / model_name / f"{model_name}_all_blurryrecons.pt"
    )
    all_predcaptions = torch.load(
        eval_path / model_name / f"{model_name}_all_predcaptions.pt"
    )
    all_clipvoxels = torch.load(
        eval_path / model_name / f"{model_name}_all_clipvoxels.pt"
    )

    # weighted averaging to improve low-level evals
    all_recons = all_recons * 0.75 + all_blurryrecons * 0.25

    # %% VISUALIZE RECONSTRUCTIONS
    if plot:
        # Testing the following images: 2 / 117 / 231 / 164 / 619 / 791

        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        jj = -1
        kk = 0
        for j in np.array([2, 165, 119, 619, 231, 791]):
            jj += 1
            axes[kk][jj].imshow(utils.torch_to_Image(all_images[j]))
            axes[kk][jj].axis("off")
            axes[kk][jj].set_title(
                wrap_title(str(all_captions[[j]]), wrap_width=30), fontsize=8
            )
            jj += 1
            axes[kk][jj].imshow(utils.torch_to_Image(all_recons[j]))
            axes[kk][jj].axis("off")
            axes[kk][jj].set_title(
                wrap_title(str(all_predcaptions[[j]]), wrap_width=30), fontsize=8
            )
            if jj == 3:
                kk += 1
                jj = -1

    # %% BRAIN <-> IMAGE RETRIEVAL
    if True:
        print("Brain <-> Image Retrieval")
        clip_img_embedder.to(device)
        percent_correct_fwds, percent_correct_bwds = [], []
        percent_correct_fwd, percent_correct_bwd = None, None

        with torch.cuda.amp.autocast(dtype=torch.float16):
            for test_i, loop in enumerate(tqdm(range(30), leave=False)):
                random_samps = np.random.choice(
                    np.arange(len(all_images)), size=300, replace=False
                )
                emb = clip_img_embedder(
                    all_images[random_samps].to(device)
                ).float()  # CLIP-Image

                emb_ = all_clipvoxels[random_samps].to(device).float()  # CLIP-Brain

                # flatten if necessary
                emb = emb.reshape(len(emb), -1)
                emb_ = emb_.reshape(len(emb_), -1)

                # l2norm
                emb = nn.functional.normalize(emb, dim=-1)
                emb_ = nn.functional.normalize(emb_, dim=-1)

                labels = torch.arange(len(emb)).to(device)
                bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)  # clip, brain
                fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)  # brain, clip

                assert len(bwd_sim) == 300

                percent_correct_fwds = np.append(
                    percent_correct_fwds, utils.topk(fwd_sim, labels, k=1).item()
                )
                percent_correct_bwds = np.append(
                    percent_correct_bwds, utils.topk(bwd_sim, labels, k=1).item()
                )

        percent_correct_fwd = np.mean(percent_correct_fwds)
        fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
        fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)
        result.append(percent_correct_fwd)
        result.append(fwd_ci)

        percent_correct_bwd = np.mean(percent_correct_bwds)
        bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
        bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)
        result.append(percent_correct_bwd)
        result.append(bwd_ci)

        # Move model back to CPU so that the rest of the test can run
        clip_img_embedder.cpu()
        torch.cuda.empty_cache()
    # %%
    if plot:
        fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(11, 12))
        for trial in range(4):
            ax[trial, 0].imshow(utils.torch_to_Image(all_images[random_samps][trial]))
            ax[trial, 0].set_title("original\nimage")
            ax[trial, 0].axis("off")
            for attempt in range(5):
                which = np.flip(np.argsort(fwd_sim[trial].cpu().numpy()))[attempt]
                ax[trial, attempt + 1].imshow(
                    utils.torch_to_Image(all_images[random_samps][which].cpu())
                )
                ax[trial, attempt + 1].set_title(f"Top {attempt+1}")
                ax[trial, attempt + 1].axis("off")
        fig.tight_layout()
        plt.show()

    # %% PIXCORR
    print("pixcorr")
    # Flatten images while keeping the batch dimension
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess(all_recons).view(len(all_recons), -1).cpu()

    corrsum = 0
    for i in tqdm(range(len(all_images)), leave=False):
        corr = np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
        if not np.isnan(corr):
            corrsum += corr
    corrmean = corrsum / len(all_images)

    pixcorr = corrmean
    result.append(pixcorr)
    # %% SSIM
    print("ssim")
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_recons).permute((0, 2, 3, 1)).cpu())

    ssim_score = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), leave=False):
        ssim_score.append(
            ssim(
                rec,
                im,
                multichannel=True,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )

    ssim_mean = np.mean(ssim_score)
    result.append(ssim_mean)
    # %% ALEXNET
    if True:
        print("alexnet")
        alex_model.to(device)

        # see alex_weights.transforms()
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Top-2
        all_per_correct = two_way_identification(
            all_recons.to(device).float(), all_images, alex_model, preprocess, "features.4"
        )
        alexnet2 = np.mean(all_per_correct)
        result.append(alexnet2)

        # Top-5
        all_per_correct = two_way_identification(
            all_recons.to(device).float(), all_images, alex_model, preprocess, "features.11"
        )
        alexnet5 = np.mean(all_per_correct)
        result.append(alexnet5)

        # Move model back to CPU so that the rest of the test can run
        alex_model.cpu()
        torch.cuda.empty_cache()
    # %% INCEPTIONV3
    print("InceptionV3")
    inception_model.to(device)

    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_per_correct = two_way_identification(
        all_recons, all_images, inception_model, preprocess, "avgpool"
    )

    inception = np.mean(all_per_correct)
    result.append(inception)

    # Move model back to CPU so that the rest of the test can run
    inception_model.cpu()
    torch.cuda.empty_cache()
    # %% CLIP
    print("CLIP")
    clip_model.to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    all_per_correct = two_way_identification(
        all_recons, all_images, clip_model.encode_image, preprocess, None
    )  # final layer
    clip_ = np.mean(all_per_correct)
    result.append(clip_)

    # Move model back to CPU so that the rest of the test can run
    clip_model.cpu()
    torch.cuda.empty_cache()
    # %% EFFICIENTNET
    print("EfficientNet")
    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = eff_model(preprocess(all_images))["avgpool"]
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))["avgpool"]
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    effnet = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    result.append(effnet)
    # %% SWAV
    print("SwAV")
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = swav_model(preprocess(all_images))["avgpool"]
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = swav_model(preprocess(all_recons))["avgpool"]
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    swav = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    result.append(swav)
    # %%
    results.append(result)
    print(results)
    pd.DataFrame(results, columns=column_names).to_csv(eval_path / model_name / f"{model_name}_results.csv")
