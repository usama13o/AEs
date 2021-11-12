# Standard libraries
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
import pywick.transforms.tensor_transforms as ts

import seaborn as sns
import matplotlib
from matplotlib.colors import to_rgb
import os
import json
import math
from posixpath import split
import numpy as np
from pytorch_lightning.loggers import WandbLogger

# Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from data_load import glas_dataset, test_peso
from helpers import GenerateCallback
from models import Autoencoder
from utils import Resize
set_matplotlib_formats('svg', 'pdf')  # For export
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()

# Progress bar

# PyTorch
# Torchvision
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    import pytorch_lightning as pl

# Tensorboard extension (for visualization purposes later)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "F:\\Data\\test\\train\\cls2\\"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


# Get data
# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([
    Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur((3, 3)),
    ]), p=0.3),
    ts.ChannelsFirst(),
    ts.TypeCast(['float', 'float']),
    ts.ChannelsLast(),
    # ts.AddChannel(axis=0),
    ts.TypeCast(['float', 'long']),
])

# Loading the training dataset. We need to split it into a training and validation part
pl.seed_everything(42)

# Loading the test set
train_dataset = glas_dataset(
    root_dir=DATASET_PATH, split='train', transform=transform)
valid_dataset =glas_dataset(
    root_dir=DATASET_PATH, split='valid', transform=transform)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=128,
                               shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
val_loader = data.DataLoader(
    valid_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=8)


def get_train_images(num):
    return torch.stack([train_dataset[i] for i in range(num)], dim=0)


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack(
        [img1, img2], dim=0), nrow=2, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


# for i in range(1):
#     # Load example image
#     img, _ = train_dataset[i]
#     img_mean = img.mean(dim=[1, 2], keepdims=True)

#     # Shift image by one pixel
#     SHIFT = 1
#     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
#     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
#     img_shifted[:, :1, :] = img_mean
#     img_shifted[:, :, :1] = img_mean
#     compare_imgs(img, img_shifted, "Shifted -")

#     # Set half of the image to zero
#     img_masked = img.clone()
#     img_masked[:, :img_masked.shape[1]//2, :] = img_mean
#     compare_imgs(img, img_masked, "Masked -")
WANDB_API_KEY="4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb_logger = False
# Training ########## Training
def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    # wandb_logger = WandbLogger(name=f'{latent_dim}_',project='AutoEPI')
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"glass_{latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=1000,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(
                                        get_train_images(8), every_n_epochs=1),
                                    LearningRateMonitor("epoch")],
                                    logger=wandb_logger if wandb_logger else True)
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, f"glass_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=128, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(
        model, test_dataloaders=val_loader, verbose=False)
#     test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"val": val_result}
    return model, result


model_dict = {}
for latent_dim in [2048]:
    model_ld, result_ld = train_cifar(latent_dim)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

latent_dims = sorted([k for k in model_dict])
val_scores = [model_dict[k]["result"]["val"][0]["test_loss"]
              for k in latent_dims]

# fig = plt.figure(figsize=(6,4))
# plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
# plt.xscale("log")
# plt.xticks(latent_dims, labels=latent_dims)
# plt.title("Reconstruction error over latent dimensionality", fontsize=14)
# plt.xlabel("Latent dimensionality")
# plt.ylabel("Reconstruction error")
# plt.minorticks_off()
# plt.ylim(0,100)
# plt.show()
