from datetime import datetime
import random

import numpy as np
from datasets import svs_h5_dataset
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
DATASET_PATH = r"/home/uz1/data/tupa/patches"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
IMG_SIZE =224


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
# train_dataset = glas_dataset(
#     root_dir=DATASET_PATH, split='all', transform=transform)
# valid_dataset = glas_dataset(
#     root_dir=DATASET_PATH, split='valid', transform=transform)
train_dataset = svs_h5_dataset(DATASET_PATH,transform=transform)
# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_dataset, batch_size=16,
                               shuffle=False, drop_last=True, pin_memory=False, num_workers=4)


def get_train_images(num):
    return torch.stack([train_dataset[i] for i in range(num)], dim=0)


# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = r"/mnt/data/Other/DOWNLOADS/epoch=499-step=48499.ckpt"
pretrained_filename = r"/mnt/data/Other/DOWNLOADS/epoch=999-step=27999 (1).ckpt"
pretrained_filename = r"/home/uz1/saved_models_AE/AE_SVS_512.ckpt/lightning_logs/version_0/checkpoints/epoch=14-step=12569.ckpt"
model = Autoencoder(base_channel_size=64, latent_dim=512,
                    width=IMG_SIZE, height=IMG_SIZE)
model = Autoencoder.load_from_checkpoint(pretrained_filename)
# load_moco_checkpoint(model.encoder,pretrained_filename)

# We use the following model throughout this section.
# If you want to try a different latent dimensionality, change it here!


def embed_imgs(model, data_loader,lim=1000):
    # Encode all images in the data_laoder using model, and return both images and encodings
    if lim==None:
        lim=1000000000

    targ_list , img_list, embed_list = [], [],[]
    model.eval()
    max = 0
    for i in tqdm(range(len(train_dataset)), desc="Encoding images", leave=False, total=len(train_dataset)):
        idx = random.randint(0,len(train_dataset))
        imgs,targ = train_dataset[idx]
        targ_list.append(targ)
        max += 1
        with torch.no_grad():
            # print("Encoding image")
            z = model.encoder(imgs.to(model.device).unsqueeze(0))
        img_list.append(imgs)

        embed_list.append(z)
        if max >= lim:
            return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0),torch.cat(targ_list, dim=0))
print('Loaded and ready to encode images !!')

def embed_imgs_labels(model, data_loader,kmean):
    # Encode all images in the data_laoder using model, and return both images and encodings

    targ_list , img_list, embed_list = [], [],[]
    model.eval()
    for imgs,targ in tqdm(data_loader, desc="Encoding images", leave=False, total=len(data_loader)):

        with torch.no_grad():
            # print("Encoding image")
            z = model.encoder(imgs.to(model.device))
            y = kmean.predict(z)
            targ_list.append(y)
        embed_list.append(z)
    return np.stack(targ_list,0).reshape(-1)
if __name__ == "__main__":
    print("Encoding ..")
    train_img_embeds = embed_imgs(model, train_loader,6000)
    # train_img_embeds  = embed_imgs(model, val_loader)


    


    # Plot the closest images for the first N test images as example
    # for i in [25,  15, 16, 17, 24]:
    #     find_similar_images(
    #         train_img_embeds[0][i], train_img_embeds[1][i], key_embeds=train_img_embeds,knn=True,dist_metric='cdist')



    # show_closest_images(train_img_embeds)

    # Create a summary writer
    writer = SummaryWriter(CHECKPOINT_PATH)

    # Note: the embedding projector in tensorboard is computationally heavy.
    # Reduce the image amount below if your computer struggles with visualizing all 10k points
    NUM_IMGS = 1000
    print(NUM_IMGS)
    cluster = True
    if cluster:

        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist

        #find best K 

        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = range(1, 10)
        X = train_img_embeds[1]
        
        for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)
        
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeanModel.inertia_)
        
            mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = kmeanModel.inertia_
        



        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.savefig('./The Elbow Method using Distortion.png')

        plt.close()
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.savefig('./The Elbow Method using Inertia.png')


        kmeans = KMeans(n_clusters=3, random_state=0).fit(train_img_embeds[1])
        print(len(kmeans.labels_))
        targs = embed_imgs_labels(model,train_loader,kmeans)
        np.save("saved_labels_svs",targs)
        print("Size of labels is: ", targs.shape)
        import joblib 
        joblib.dump(kmeans,"kmeans_class.pkl")

        # create_stitched_image((train_img_embeds[0][:NUM_IMGS]+1), train_img_embeds[1][:NUM_IMGS],kmeans.labels_)


