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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import seaborn as sns
import matplotlib
from matplotlib.colors import to_rgb
import os
import json
import math
from posixpath import split
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

# Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from data_load import glas_dataset, test_peso
from datasets import GeoFolders_2, svs_h5_dataset
from helpers import GenerateCallback, GenerateCallback_Single_images
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


from datasets import combined_medinst_dataset
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# DATASET_PATH = "F:\\Data\\test\\train\\cls2\\"
DATASET_PATH = "/home/uz1/data/tupa/patches"
DATASET_PATH = "/home/uz1/DATA!/medmnist"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models_AE/"
n_epochs=40
bs=128
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
    transforms.Resize((224,224)) ,
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
#train_dataset = svs_h5_dataset(
#    root_dir=DATASET_PATH, split='train', transform=transform)
#valid_dataset =svs_h5_dataset(
#    root_dir=DATASET_PATH, split='valid', transform=transform)
train_dataset = combined_medinst_dataset(root=DATASET_PATH,transform=transform)
valid_dataset= combined_medinst_dataset(root=DATASET_PATH,split='val',transform=transform)
# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=bs,
                               shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
val_loader = data.DataLoader(
    valid_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=8)


def get_train_images(num):
    print(train_dataset[1][0].shape)
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


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


WANDB_API_KEY="4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb_logger = False
# Training ########## Training
def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    model_path = f"AE_SVS_{latent_dim}.ckpt"
    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, model_path)
    # wandb_logger = WandbLogger(name=f'{latent_dim}_',project='AutoEPI')
    trainer = pl.Trainer(default_root_dir=pretrained_filename,
                         gpus=[0,1] if str(device).startswith("cuda") else 0,
                         max_epochs=n_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback_Single_images(
                                        get_train_images(8), every_n_epochs=1),
                                    LearningRateMonitor("epoch"),
                                    # EarlyStopping(monitor="val_loss",patiance =5)
                                    ],
                                    logger=wandb_logger if wandb_logger else True,
                                    plugins=DDPPlugin(find_unused_parameters=False),

                                    )
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=64, latent_dim=latent_dim,steps=len(train_loader),n_epochs=n_epochs)
        trainer.fit(model, train_loader, val_loader)
    
    
    # # Test best model on validation and test set
    val_result = trainer.test(
        model, test_dataloaders=val_loader, verbose=False)
#     test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"val": val_result}
    return model, result


model_dict = {}
for latent_dim in [512]:
    model_ld, result_ld = train_cifar(latent_dim)
    # model_ld.save(f'{latent_dim}_.h5')
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

# latent_dims = sorted([k for k in model_dict])
# val_scores = [model_dict[k]["result"]["val"][0]["test_loss"]
#               for k in latent_dims]

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
