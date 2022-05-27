import math
import pandas as pd
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

from utils import open_image_np, open_target_get_class 
from datasets import glas_dataset,GeoFolders,GeoFolders_2
def open_pickled_file(fn):
  with open(fn, "rb") as f_in:
    arr_new = pickle.load(f_in)
  return arr_new






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
kfold=False
DATASET_PATH ="/home/uz1/data/geo/slices/64"
# DATASET_PATH ="/home/uz1/data/geo/slices/geo2_slices_pil/geo2/64"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH =  "./saved_models_3dAE_NEW_DATA_TESTING_RES/"
# Setting the seed
# Loading the test set
from torchvision.datasets import ImageFolder
# train_dataset = glas_dataset(
#     root_dir=DATASET_PATH, split='train', transform=transform)
# valid_dataset =glas_dataset(
    # root_dir=DATASET_PATH, split='valid', transform=transform)


ress = []
for p in list(range(19))[:]:
    print(f"Picking {p} . . .")
    model_path = str(sorted(list(list(Path('/home/uz1/saved_models_3dAE_NEW_DATA/').glob(f'./*__{p}*'))[1].glob('./*/*/*/*')))[1])
    train_dataset = GeoFolders_2(
        root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/',pick=p)# picks all but (p)
    valid_dataset = GeoFolders_2(
        root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/',split="valid",pick=p)# valid isolates (p) region

    print(f"Training on {len(train_dataset.list_regions)}: {train_dataset.list_regions}")
    print(f"\n  Validation on {len(valid_dataset.list_regions)}: {valid_dataset.list_regions}")
    print('Path to model: ',model_path)

    #find labels of the dataset

    # train_dataset = GeoFolders(
        # root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/geo2_unclipped/0/',balance=args.balance,k_labels_path="/home/uz1/k_labels_.pickle")
    # train_dataset[100]
    # valid_dataset= GeoFolders(
        # root='/home/uz1/data/geo/slices/geo1_slices_pil/geo1/64',  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/0')
    #
    from sklearn.model_selection import StratifiedShuffleSplit
   
        
    splits = StratifiedShuffleSplit(2)

    get_labels = lambda a,i: [int(x[1]) for x in np.array(a.samples)[i]]
            
        #  We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_dataset, batch_size=128,
                                shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
    val_loader = data.DataLoader(
        valid_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=8,sampler = None )



    def get_train_images(num):
        from itertools import islice
        filtered = (x for x in valid_dataset if x[1] == 1)
        filtered_ = (x for x in valid_dataset if x[1] == 0)
        # filtered__ = (x for x in train_dataset if x[1] == 3)
        b = list(islice(filtered, int(num/3)))
        a = list(islice(filtered_, int(num/3)))
        # v = list(islice(filtered__, int(num/3)))
        b.extend([*a])
        # b.extend([*v])
        return b



    # wandb_logger = WandbLogger(name=f'{latent_dim}_',project='AutoEPI')
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"3DAE_test__{p}.ckpt"),
                            gpus=[0] if str(device).startswith("cuda") else 0,
                            max_epochs=1,
                            limit_train_batches=0,
                            limit_val_batches=0,
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



   
    model = deeplab(num_classes=9,proj_output_dim=1024,pred_hidden_dim=512,num_ch=9,num_predicted_clases=2)
    model =deeplab.load_from_checkpoint(model_path)
    trainer.fit(model, train_loader)
    # res = trainer.test(model, train_loader)
    res = trainer.test(model,val_loader)
    res[0]['pick'] = valid_dataset.list_regions
    ress.append(res)

    #save each run as row in a dataframe
    '''
    Make a dataframe with the results
    For each entry in ress, we need to add its content to the dataframe
    '''
    all_Res = pd.DataFrame({})
    for res in ress:
        res_df = pd.DataFrame(res)
        all_Res = all_Res.append(res_df)
    all_Res.to_csv(f"{CHECKPOINT_PATH}/3DAE_test_results.csv")


