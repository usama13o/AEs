import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from os.path import basename
from utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np, open_target_np_glas, open_target_np_peso;                   
from glob import glob 
import random
from tqdm import tqdm 

class glas_dataset(data.Dataset):
    def filter_size(self,test_size,filenames):
        list_of_idxs = []
        for idx,file in tqdm(enumerate(filenames)):
            try:
                temp_img = open_image_np(file).shape
            except:
                print(file)
                list_of_idxs.append(idx)
            if temp_img[0] == test_size[0] and temp_img[1] == test_size[1]:
                continue 
            else:
                list_of_idxs.append(idx)
        filenames = np.delete(filenames,list_of_idxs)
                
        return filenames

    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(glas_dataset, self).__init__()
        img_dir= root_dir
        # targets are a comob of two dirs 1- normal 1024 patches 2- Tum 1024
        self.image_filenames  = sorted(glob(img_dir+'/*'))
        #filter the iamges with a different size than the required one 
        # self.image_filenames = self.filter_size(test_size,self.image_filenames)
        
        sp= self.image_filenames.__len__()
        sp= int(train_pct *sp)
        # random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        
        if split == 'all':
            self.image_filenames= self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]

            # find the mask for the image

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input  = open_image_np(self.image_filenames[index])
            if input.shape[2] > 3:
                input = input[:,:,0:3]
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        if index >=1700 and index < 1800:
            targ = 1
        else:
            targ = 0

        return (input,targ)

    def __len__(self):
        return len(self.image_filenames)




class test_peso(data.Dataset):

    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(test_peso, self).__init__()
        img_dir= root_dir
        # targets are a comob of two dirs 1- normal 1024 patches 2- Tum 1024
        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if is_image_file(x)])
        sp= self.image_filenames.__len__()
        sp= int(train_pct *sp)
        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input  = open_image_np(self.image_filenames[index])
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        return input

    def __len__(self):
        return len(self.image_filenames)
