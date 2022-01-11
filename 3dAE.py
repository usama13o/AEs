import torch
from random import sample
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

from helpers import GenerateCallback, HookBasedFeatureExractorCallback
from deeplab import deeplab

from unet_3D import unet_3D
from torchsampler import ImbalancedDatasetSampler


from argumentparser import args

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

    def __init__(self,root,transform,raw_dir,balance=True,k_labels_path=None):
        self.raw_dir = raw_dir 
        self.raw_dir = Path(self.raw_dir)
        super(GeoFolders,self).__init__(root,transform)
        self.k_labels_path = k_labels_path if k_labels_path is not None else None 
        if balance:
            self.samples = [x for x in self.samples if x[1]==1]
            self.samples.extend(sample([x for x in self.imgs if x[1] ==0],1000))
        else:
            self.samples=self.imgs
            self.samples = (sorted(self.samples,key=lambda x: int(Path(x[0]).stem)))
        print("Current smaple count ", len(self.samples))
    def __getitem__(self, index: int):

        path, target = self.samples[index]
        path = Path(path)
        #search for the equaivlant tile in the raw tiles
        path= self.raw_dir.joinpath(f'{path.stem}.pickle')

        #return actula color tile fro testing 
        color_path = np.array(PIL.Image.open(self.samples[index][0]))

        input  = open_pickled_file(path)
        k_label = open_pickled_file(self.k_labels_path)[index] if self.k_labels_path is not None else None
        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)
            # color_path=self.transform(color_path)

        if k_label is not None:
            # print(str(target),"mapped to ",str(k_label)) # 0 (0,2) | 1 (1,3)
            return input,k_label,color_path
        return input,target,color_path







# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([
    # Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    ts.TypeCast(['float', 'float']),
    ts.StdNormalize(),
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
DATASET_PATH ="/home/uz1/data/geo/slices/geo2_slices_pil/geo2/64"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH =  "./saved_models_3dAE/"
# Setting the seed
# Loading the test set
from torchvision.datasets import ImageFolder
# train_dataset = glas_dataset(
#     root_dir=DATASET_PATH, split='train', transform=transform)
# valid_dataset =glas_dataset(
    # root_dir=DATASET_PATH, split='valid', transform=transform)

train_dataset = GeoFolders(
    root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/geo2_unclipped/0/',balance=args.balance,k_labels_path="/home/uz1/k_labels_.pickle")

# valid_dataset= GeoFolders(
    # root='/home/uz1/data/geo/slices/geo1_slices_pil/geo1/64',  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/0')
#
from sklearn.model_selection import StratifiedShuffleSplit
def try_my_operation(item): return item[1]
if kfold:
    print('Listing Indices for Targets . . . ')
    executor = concurrent.futures.ProcessPoolExecutor(20)
    futures = [executor.submit(try_my_operation, item) for item in train_dataset]
    concurrent.futures.wait(futures)
    list_targs = [x.result() for x in futures]
else:
    list_targs=np.zeros(len(train_dataset))
    
splits = StratifiedShuffleSplit(10)

get_labels = lambda a,i: [int(x[1]) for x in np.array(a.samples)[i]]
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)),list_targs)):
    train_sampler = ImbalancedDatasetSampler(train_dataset,train_idx)
    test_sampler = ImbalancedDatasetSampler(train_dataset,val_idx)

    ss=[]
    #counts class dist
    for idx,val in enumerate(list(train_sampler)):
        ss.append(list_targs[val])
    

    print(f"** No. of train samples for fold({fold}) is {len(train_idx)} - 0/1 = {str(np.unique(ss,return_counts=True)[1])}")
    print(f"** No. of validation samples for fold({fold}) is {len(val_idx)}")
        
    #  We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_dataset, batch_size=128,
                                shuffle=False, drop_last=True, pin_memory=False, num_workers=8,sampler=train_sampler if train_sampler else None)
    val_loader = data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=8,sampler = test_sampler if test_sampler else None )



    def get_train_images(num):
        from itertools import islice
        filtered = (x for x in train_dataset if x[1] == 1)
        filtered_ = (x for x in train_dataset if x[1] == 0)
        b = list(islice(filtered, int(num/2)))
        a = list(islice(filtered_, int(num/2)))
        b.extend([*a])
        return b



    # wandb_logger = WandbLogger(name=f'{latent_dim}_',project='AutoEPI')
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"3DAE_{fold}_{args.tag}.ckpt"),
                            gpus=[1] if str(device).startswith("cuda") else 0,
                            max_epochs=1000,
                            callbacks=[ModelCheckpoint(save_top_k=2,monitor='class_loss_val',save_weights_only=True),
                                    GenerateCallback(
                                        get_train_images(12), every_n_epochs=1),
                                    LearningRateMonitor("epoch"),
                                    EarlyStopping(monitor="class_loss_val",patience=50,verbose=True),
                                    # HookBasedFeatureExractorCallback()
                                    ],
                            log_every_n_steps=1        
                                    )
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None



    # pretrained_filename = next(Path('/home/uz1/saved_models_3dAE/3DAE_0.ckpt/lightning_logs/').joinpath(f"version_{len(list(Path('/home/uz1/saved_models_3dAE/3DAE_0.ckpt/lightning_logs/').glob('version_*')))-1}/checkpoints/").glob('*'))
    # # Check whether pretrained model exists. If yes, load it and skip training
    # if os.path.isfile(pretrained_filename) and resume ==True:
    #     print("Found pretrained model, loading...")
    #     try:
    #         # assert 1==0
    #         model = unet_3D.load_from_checkpoint(pretrained_filename.__str__())
    #     except:
    #         model = unet_3D(n_classes=1,in_channels=1,proj_output_dim=1024,pred_hidden_dim=512)
    model = deeplab(num_classes=9,proj_output_dim=1024,pred_hidden_dim=512,num_ch=9)
    trainer.fit(model, train_loader,val_loader)


