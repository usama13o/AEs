from datetime import datetime
import random
from utils import Resize
from models import Autoencoder
from helpers import GenerateCallback
from data_load import glas_dataset, test_peso
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from posixpath import split
from matplotlib.colors import to_rgb
import matplotlib
import seaborn as sns
from PIL import ImageDraw, ImageFont
import pywick.transforms.tensor_transforms as ts
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard._embedding import make_sprite
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models import load_moco_checkpoint
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Imports for plotting

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
# DATASET_PATH = "/mnt/data/Other/DOWNLOADS/WSIData/filtered/PNG/train/"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
DATASET_PATH = r"/mnt/data/Other/DOWNLOADS/slices (3)/slices/0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
IMG_SIZE = 128


global now
now = datetime.now()  # current date and time
def create_stitched_image(images, embeds,labels):
    print("images have the shape : ", images.shape)
    colors = [

        (255, 255, 255),
        (0, 98, 255),
        (229, 255, 0),
        (255, 0, 255),
        (255, 55, 0),
        (255, 255, 0),
        (24, 55, 0),
        (155, 0, 0),
    ]
    from glob import glob
    import numpy as np
    import PIL
    # images = images.detach().permute(0,2,3,1).numpy()
    # stand_image_shape = images[0].shape

    # make_sprite(images,save_path='./')
    mod_images = []
    for idx, im in enumerate(images):
        colour = colors[labels[idx]]
        im = im.cpu().permute(1, 2, 0).numpy()
        im = np.uint8(im * 255).clip(0, 255)
        # im = np.uint8(im)
        im = PIL.Image.fromarray(im)
        overlay = ImageDraw.Draw(im)
        
        overlay.rectangle((0, 0, im.size[0], im.size[1]),
                          fill=None,
                          outline=colour, width=5)

        mod_images.append(np.array(im))
    mod_images = torch.Tensor(mod_images).permute(0, 3, 1, 2)
    del images
    print('Making sprite image: ', mod_images.shape)
    make_sprite(mod_images, save_path='./')
    writer.add_embedding(embeds,  # Encodings per image
                     label_img=mod_images, global_step=now.strftime("%m_%d_%Y__%H_%M_%S"))# Adding the original images to the plot



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
    # Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE)),
    ts.ChannelsFirst(),
    ts.TypeCast(['float', 'float']),
    ts.ChannelsLast(),
    ts.TypeCast(['float', 'long']),
])

# Loading the training dataset. We need to split it into a training and validation part
pl.seed_everything(42)

# Loading the test set
train_dataset = glas_dataset(
    root_dir=DATASET_PATH, split='all', transform=transform)
valid_dataset = glas_dataset(
    root_dir=DATASET_PATH, split='valid', transform=transform)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=6,
                               shuffle=False, drop_last=True, pin_memory=False, num_workers=4)
val_loader = data.DataLoader(
    valid_dataset, batch_size=6, shuffle=False, drop_last=False, num_workers=4)


def get_train_images(num):
    return torch.stack([train_dataset[i] for i in range(num)], dim=0)


# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = r"/mnt/data/Other/DOWNLOADS/epoch=499-step=48499.ckpt"
pretrained_filename = r"/mnt/data/Other/DOWNLOADS/epoch=999-step=27999 (1).ckpt"
model = Autoencoder(base_channel_size=128, latent_dim=128,
                    width=IMG_SIZE, height=IMG_SIZE)
model = Autoencoder.load_from_checkpoint(pretrained_filename)
# load_moco_checkpoint(model.encoder,pretrained_filename)

# We use the following model throughout this section.
# If you want to try a different latent dimensionality, change it here!


def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings

    targ_list , img_list, embed_list = [], [],[]
    model.eval()
    max = 0
    for imgs,targ in tqdm(data_loader, desc="Encoding images", leave=False, total=len(data_loader)):

        targ_list.append(targ)
        max += 1
        with torch.no_grad():
            # print("Encoding image")
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)

        embed_list.append(z)
        if max >= 1000:
            return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0),torch.cat(targ_list, dim=0))
print('Loaded and ready to encode images !!')
if __name__ == "__main__":
    print("Encoding ..")
    train_img_embeds = embed_imgs(model, train_loader)
    # train_img_embeds  = embed_imgs(model, val_loader)


    def find_similar_images(query_img, query_z, key_embeds, K=8,knn=False,dist_metric='cos'):
        # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
        if dist_metric== 'cos':
            dist = torch.cosine_similarity(query_z[None, :], key_embeds[1])
        else:
            dist = torch.cdist(query_z[None, :], key_embeds[1],p=2)
        dist = dist.squeeze(dim=0)
        if knn:
            from sklearn.neighbors import NearestNeighbors 
            neigh = NearestNeighbors(n_neighbors=8)
            nn = neigh.fit(key_embeds[1])
            dist, indices = nn.kneighbors(query_z.reshape(1,-1))
            indices = indices.reshape(-1)
        else:
            dist, indices = torch.sort(dist)
        # Plot K closest images
        imgs_to_display = torch.cat(
            [query_img[None], key_embeds[0][indices[:K]]], dim=0)
        grid = torchvision.utils.make_grid(
            imgs_to_display, nrow=K+1, normalize=True, range=(-1, 1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(12, 3))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'grid_{indices[1]}_{dist_metric}_knn__{str(knn)}.png')
        # plt.show()


    # Plot the closest images for the first N test images as example
    for i in [25,  15, 16, 17, 24]:
        find_similar_images(
            train_img_embeds[0][i], train_img_embeds[1][i], key_embeds=train_img_embeds,knn=True,dist_metric='cdist')

    def show_closest_images(train_img_embeds,k=50):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        neigh = NearestNeighbors(n_neighbors=k)
        nn = neigh.fit(train_img_embeds[1])
        ind_list=[]
        for i in range(1700,1800):
            print(i)
            _,ind = nn.kneighbors(train_img_embeds[1][i].reshape(1,-1))
            ind_list.extend(*ind)
        topk = np.unique(ind_list,return_counts=True)[0][np.argpartition(np.unique(ind_list,return_counts=True)[1], -4)[-4:]]
        print("topk --> ",topk)
        topk_imgs=torch.cat([train_img_embeds[0][topk]],dim=0)
        img_g = torchvision.utils.make_grid(topk_imgs,nrow=4,normalize=True,range=(-1,1)).permute(1,2,0)
        plt.figure(figsize=(12, 3))
        plt.imshow(img_g)
        plt.axis('off')

    show_closest_images(train_img_embeds)
    # Create a summary writer
    writer = SummaryWriter(CHECKPOINT_PATH)

    # Note: the embedding projector in tensorboard is computationally heavy.
    # Reduce the image amount below if your computer struggles with visualizing all 10k points
    NUM_IMGS = 1000
    print(NUM_IMGS)
    cluster = True
    if cluster:

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=4, random_state=0).fit(train_img_embeds[1])
        print(len(kmeans.labels_))

        create_stitched_image((train_img_embeds[0][:NUM_IMGS]+1), train_img_embeds[1][:NUM_IMGS],kmeans.labels_)


