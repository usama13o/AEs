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
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from data_load import glas_dataset,test_peso
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
DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
DATASET_PATH = r"C:\Users\Usama\AEs\slices"
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


def create_stitched_image(images,labels):
    from glob import glob
    import numpy as np
    import  PIL 
    stand_image_shape = np.array(PIL.Image.open(images[0])).shape
    n_r = 6
    n_c = 7

    re_Stit = np.zeros((984,1092,stand_image_shape[2]))
    r1=0
    r2=stand_image_shape[0]
    c1=0
    c2=stand_image_shape[1]

    for im in images:
        print(im.shape)
        img = PIL.Image.open(im)
        print(r1,r2,c1,c2)

        re_Stit[r1:r2,c1:c2,:] = img

        c2 = c2 + stand_image_shape[1]
        c1 = c1 + stand_image_shape[1]
        if c1 >= stand_image_shape[1] * n_c:
            c1=0
            c2=156
            r2 = r2 + stand_image_shape[0]
            r1 = r1 + stand_image_shape[0]


    PIL.Image.fromarray(re_Stit.astype(np.uint8)).save("restitched.png")

# Get data
# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([
                                Resize((128,128)),
                                transforms.ToTensor(),
                                ts.ChannelsFirst(),
                                ts.TypeCast(['float', 'float']),
                                ts.ChannelsLast(),
                                # ts.AddChannel(axis=0),
                                ts.TypeCast(['float', 'long']),
                                ])

# Loading the training dataset. We need to split it into a training and validation part
pl.seed_everything(42)

# Loading the test set
train_dataset =glas_dataset(
    root_dir=DATASET_PATH, split='all', transform=transform)
valid_dataset =glas_dataset(
    root_dir=DATASET_PATH, split='valid', transform=transform)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=6,
                               shuffle=False, drop_last=True, pin_memory=False, num_workers=4)
val_loader = data.DataLoader(
    valid_dataset, batch_size=6, shuffle=False, drop_last=False, num_workers=4)


def get_train_images(num):
    return torch.stack([train_dataset[i] for i in range(num)], dim=0)

# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = r"C:\Users\Usama\saved_models\glass_1024\lightning_logs\version_0\checkpoints\epoch=499-step=13999.ckpt"
model = Autoencoder.load_from_checkpoint(pretrained_filename)


# We use the following model throughout this section. 
# If you want to try a different latent dimensionality, change it here!
# model = model_dict[128]["model"] 

def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    max=0
    for imgs in tqdm(data_loader, desc="Encoding images", leave=False):
        max+=1
        with torch.no_grad():
                # print("Encoding image")
                z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        
        embed_list.append(z)
        if max >=1000:
                return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))

train_img_embeds = embed_imgs(model, train_loader)
test_img_embeds = embed_imgs(model, val_loader)


def find_similar_images(query_img, query_z, key_embeds, K=8):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None,:], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K+1, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12,3))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# Plot the closest images for the first N test images as example
# for i in range(8):
    # find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)

# Create a summary writer
writer = SummaryWriter(CHECKPOINT_PATH)

# Note: the embedding projector in tensorboard is computationally heavy.
# Reduce the image amount below if your computer struggles with visualizing all 10k points
NUM_IMGS = len(train_dataset) 
print(NUM_IMGS)


from sklearn.cluster import KMeans

from datetime import datetime

now = datetime.now() # current date and time
kmeans = KMeans(n_clusters=4, random_state=0).fit(train_img_embeds[1])
print(len(kmeans.labels_))

create_stitched_image(train_img_embeds[0],kmeans.labels_)



writer.add_embedding(train_img_embeds[1][:NUM_IMGS], # Encodings per image
                     metadata=[str(i) for i in kmeans.labels_ ], # Adding the labels per image to the plot
                     label_img=(train_img_embeds[0][:NUM_IMGS]+1)/2.0,global_step=now.strftime("%m_%d_%Y__%H_%M_%S")) # Adding the original images to the plot

