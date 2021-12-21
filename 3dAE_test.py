import torch
import PIL
from torch import nn
from torch.autograd import Variable
import os
import tiffile
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.data as data
import pickle
import glob
import pytorch_lightning as pl
import concurrent
from pathlib import Path
# Standard libraries
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
import torchvision
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
import pywick.transforms.tensor_transforms as ts
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from matplotlib.colors import to_rgb
import os
import numpy as np
from pytorch_lightning.loggers import WandbLogger

from helpers import GenerateCallback, GenerateTestCallback, HookBasedFeatureExractorCallback

from unet_3D import unet_3D



import datetime 

def open_pickled_file(fn):
  with open(fn, "rb") as f_in:
    arr_new = pickle.load(f_in)
  return arr_new


class glas_dataset(data.Dataset):
      

    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(glas_dataset, self).__init__()
        img_dir= root_dir
        print('Looking in ',root_dir)
        # targets are a comob of two dirs 1- normal 1024 patches 2- Tum 1024
        self.image_filenames  = sorted(glob.glob(img_dir+'/*'))

        sp= len(self.image_filenames)
        sp= int(train_pct *sp)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        
        elif split == 'all':
            self.image_filenames= self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]

            # find the mask for the image
        print(len(self.image_filenames))

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform



    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        input  = open_pickled_file(self.image_filenames[index])

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)


        return input

    def __len__(self):
        return len(self.image_filenames)



class GeoFolders(torchvision.datasets.ImageFolder):

    def __init__(self,root,transform,raw_dir,target_transform=None):
        self.raw_dir = raw_dir 
        self.raw_dir = Path(self.raw_dir)
        super(GeoFolders,self).__init__(root,transform)
        self.samples = [x for x in self.samples if x[1]==1]
        self.samples.extend([x for x in self.imgs if x[1] ==0][:1000])
        print("Current smaple count ", len(self.samples))
    def __getitem__(self, index: int):

        path, target = self.samples[index]
        path = Path(path)
        #search for the equaivlant tile in the raw tiles
        path= next(self.raw_dir.glob(f'*{path.stem}.pickle'))
        #return actula color tile fro testing 
        color_path = np.array(PIL.Image.open(self.samples[index][0]))

        input  = open_pickled_file(path)

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)


        return input,target,color_path







# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([
    # Resize((128, 128)),
    transforms.ToTensor(),
    ts.TypeCast(['float', 'float']),
    # transforms.RandomApply(torch.nn.ModuleList([
        # transforms.GaussianBlur((3, 3)),
    # ]), p=0.3),
])
# Loading the training dataset. We need to split it into a training and validation part
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# make into args 
resume=True
kfold=True
DATASET_PATH ="/home/uz1/data/geo/slices/64/geo2_org"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH =  "./saved_models_3dAE/"
# Setting the seed
# Loading the test set
from torchvision.datasets import ImageFolder
# train_dataset = glas_dataset(
    # root_dir=DATASET_PATH, split='all', transform=transform)
# valid_dataset =glas_dataset(
    # root_dir=DATASET_PATH, split='valid', transform=transform)

train_dataset = GeoFolders(
    root=DATASET_PATH,  transform=transform,raw_dir="/home/uz1/data/geo/slices/64/geo2_org_raw/0/")
train_dataset= GeoFolders(
    root='/home/uz1/data/geo/slices/geo1_slices_pil/geo1/64',  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/0')

from sklearn.model_selection import StratifiedShuffleSplit

train_loader = data.DataLoader(train_dataset, batch_size=1,
                            shuffle=False, drop_last=True, pin_memory=False, num_workers=8)



def get_train_images(num):
    b = [train_dataset[x] for x in range(num)]
    return b



trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"3DAE_TEST.ckpt"),
                        gpus=[0] if str(device).startswith("cuda") else 0,
                        max_epochs=1,
                        limit_train_batches=0,
                        limit_val_batches=0,
                        callbacks=[
                                GenerateTestCallback(
                                    get_train_images(200), every_n_epochs=1),
                                HookBasedFeatureExractorCallback()
                                ],
                        log_every_n_steps=1        
                                )
# If True, we plot the computation graph in tensorboard
trainer.logger._log_graph = True
# Optional logging argument that we don't need
trainer.logger._default_hp_metric = None


# Check whether pretrained model exists. If yes, load it and skip training
try:
    pretrained_filename = "/home/uz1/saved_models_3dAE/3DAE_1.ckpt/lightning_logs/version_6/checkpoints/epoch=34-step=279.ckpt"
except:
    pretrained_filename = False
    resume=False
if os.path.isfile(pretrained_filename) and resume ==True:
    print("Found pretrained model, loading...")
    model = unet_3D.load_from_checkpoint(pretrained_filename.__str__())
else:
    model = unet_3D(n_classes=1,in_channels=1,proj_output_dim=1024,pred_hidden_dim=512)
    


trainer.fit(model, train_loader)
trainer.test(model,train_loader)


