import math
import pandas as pd
import torch
from random import sample
import datetime
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

from helpers import GenerateTestCallback, HookBasedFeatureExractorCallback
from deeplab import deeplab

from unet_3D import unet_3D
from torchsampler import ImbalancedDatasetSampler
import logging
from loggingg import StreamToLogger
import sys
from argumentparser import args

import datetime

from utils import open_image_np, open_target_get_class
from datasets import GeoFolders_2_VALID, glas_dataset, GeoFolders, GeoFolders_2, GeoData_test_tiff


def open_pickled_file(fn):
    with open(fn, "rb") as f_in:
        arr_new = pickle.load(f_in)
    return arr_new


#logger
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#     filename="./out.log",
#     filemode='a'
# )

# stdout_logger = logging.getLogger('STDOUT')
# sl = StreamToLogger(stdout_logger, logging.INFO)
# sys.stdout = sl

# stderr_logger = logging.getLogger('STDERR')
# sl = StreamToLogger(stderr_logger, logging.ERROR)
# sys.stderr = sl

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([
    # Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    ts.TypeCast(['float', 'float']),
    ts.StdNormalize(),
    # transforms.RandomApply(torch.nn.ModuleList([
    # transforms.GaussianBlur((3, 3)),
    # ]), p=0.3),
])
transform_valid = transforms.Compose([
    # Resize((128, 128)),
    transforms.ToTensor(),
    ts.TypeCast(['float', 'float']),
    ts.StdNormalize(),
])
# Loading the training dataset. We need to split it into a training and validation part
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device(
    "cpu")
print("Device:", device)

# make into args
resume = True
kfold = False
# DATASET_PATH ="/home/uz1/data/geo/slices/64"
# raw_dir = '/home/uz1/data/geo/slices_raw/64/'
raw_dir = '/home/uz1/data/geo/slices_raw/test/'
DATASET_PATH = "/home/uz1/data/geo/slices/test/"

# DATASET_PATH ="/home/uz1/data/geo/slices/geo2_slices_pil/geo2/64"
# DATASET_PATH = "F:\\Data\\slices (3)\\slices\\0"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models_3dAE_NEW_DATA_TESTING_RES/"
# Setting the seed
# Loading the test set
from torchvision.datasets import ImageFolder
# train_dataset = glas_dataset(
#     root_dir=DATASET_PATH, split='train', transform=transform)
# valid_dataset =glas_dataset(
# root_dir=DATASET_PATH, split='valid', transform=transform)

#add logger to log stdout and stderr to txt

ress = []
'''
Run all models on Selected Region and save results in CSV file.
Select model based on most updated epoch model 
args:
    model_name: path of the model to run
    target: target region to test on
    which_fod: which fold to test on

'''
which_fold = 0 if args.which_fold == -1 else args.which_fold
#load tif
tif_file = "/home/uz1/data/geo/full_image/testTiff/Prosp1_KSA_1355_2541_368.tif"
print("**** running on fold ", which_fold)
coo = []
for which_fold in [0, 1]:
    now = datetime.datetime.now().strftime('%m-%d-%H:%M')
    for p in range(0, 18):
        print(f"Picking {p} . . .")
        model_path = str(
            sorted(
                list(
                    list(
                        Path('/home/uz1/saved_models_3dAE_NEW_DATA/').glob(
                            f'./*{which_fold}__{p}.*'))[-1].glob('./*/*/*/*')))
            [-1])
        print(model_path)
        # m_path = sorted(list(list(Path('/home/uz1/saved_models_3dAE_NEW_DATA/').glob(f'./*{which_fold}__{p}.*'))[-1].glob('./*/*/*/*')))
        if p in [21, 19]: p = 1
        # for model_path in m_path:
        # model_path=str(model_path)
        train_dataset = GeoData_test_tiff(root=tif_file,
                                          transform=transform,
                                          patch_size=(64, 64),limit='rotate')
        valid_dataset = GeoData_test_tiff(root=tif_file,
                                          transform=transform_valid,
                                          patch_size=(64, 64),limit='rotate')

        print('Path to model: ', model_path)

        #find labels of the dataset

        # train_dataset = GeoFolders(
        # root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/geo2_unclipped/0/',balance=args.balance,k_labels_path="/home/uz1/k_labels_.pickle")
        # train_dataset[100]
        # valid_dataset= GeoFolders(
        # root='/home/uz1/data/geo/slices/geo1_slices_pil/geo1/64',  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/0')
        #
        from sklearn.model_selection import StratifiedShuffleSplit

        splits = StratifiedShuffleSplit(2)

        get_labels = lambda a, i: [int(x[1]) for x in np.array(a.samples)[i]]

        #  We define a set of data loaders that we can use for various purposes later.
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=128,
                                       shuffle=False,
                                       drop_last=True,
                                       pin_memory=False,
                                       num_workers=8)
        val_loader = data.DataLoader(valid_dataset,
                                     batch_size=128,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=8,
                                     sampler=None)

        def get_train_images(num):
            b = list(x for x in valid_dataset)[:num]
            return b

        # extract region name for logging path of sprite image
        model_region = list(Path(DATASET_PATH).glob("*_Area_*"))
        model_region = [
            x for x in model_region if "USA_Area_3" not in str(x)
            and "Egypt" not in str(x) and "China" not in str(x)
        ]
        model_region = model_region[int(
            Path(model_path).parts[4].split("_")[-1][:-5])].parts[-1]
        # wandb_logger = WandbLogger(name=f'{latent_dim}_',project='AutoEPI')
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH,
                                          f"3DAE_test__{p}.ckpt"),
            gpus=[0] if str(device).startswith("cuda") else 0,
            max_epochs=1,
            limit_train_batches=0,
            limit_val_batches=0,
            callbacks=[
                ModelCheckpoint(save_top_k=2,
                                monitor='class_loss_val',
                                save_weights_only=True),
                # GenerateTestCallback(
                # get_train_images(len(valid_dataset)), every_n_epochs=1,logs="Per_Region_Results/"+model_region if model_path != '' else  None),
                LearningRateMonitor("epoch"),
                EarlyStopping(monitor="class_loss_val",
                              patience=50,
                              verbose=True),
                # HookBasedFeatureExractorCallback()
            ],
            log_every_n_steps=1)
        # If True, we plot the computation graph in tensorboard
        trainer.logger._log_graph = True
        # Optional logging argument that we don't need
        trainer.logger._default_hp_metric = None

        model = deeplab(num_classes=9,
                        proj_output_dim=1024,
                        pred_hidden_dim=512,
                        num_ch=9,
                        num_predicted_clases=2)
        model = deeplab.load_from_checkpoint(model_path)
        trainer.fit(model, train_loader)
        # res = trainer.test(model, train_loader)
        # res = trainer.test(model,val_loader)
        preds = []
        res = [{}]
        cooords = []
        for x in val_loader:
            x, coords = x
            # x=transform_valid(x)
            if x.isnan().all():  #or x.mean() == 0:
                continue
            _, _, class_logits = model.forward(x)
            pred = model.apply_argmax_softmax(class_logits) #.argmax(1)

            preds.append(pred)
            cooords.append(pred)
        tot_ = int(torch.stack(preds[:-1]).reshape(-1,2).argmax(1).sum())
        model_path = model_path.split('/')[4:]
        res[0]['fold_region'] = model_path[0]
        res[0]['version'] = model_path[2]
        res[0]['epoch'] = model_path[-1]
        res[0]['P'] = p
        res[0]['total_detections'] = tot_
        # res[0]['coords'] = cooords
        ress.append(res)
        tot_ = 0
        np.save(f"/home/uz1/projects/AEs/{model_path[0]}_{tot_}_{p}.npy",torch.stack(preds[:-1]).reshape(-1,2).cpu().detach().numpy())
        #save each run as row in a dataframe
        '''
        Make a dataframe with the results
        For each entry in ress, we need to add its content to the dataframe
        '''
        all_Res = pd.DataFrame({})
        for res in ress:
            res_df = pd.DataFrame(res)
            all_Res = all_Res.append(res_df)
        # save at checkpoint path and add timestamp
        print(all_Res)

        all_Res.to_csv(
            f"{CHECKPOINT_PATH}/3DAE_test_results_{args.target}_f{which_fold}_{now}.csv"
        )  #add date_to path
# np.save("./saved_coords.npy",coo)