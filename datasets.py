from bisect import bisect_right
import datetime
import glob
import math
from os import listdir
from random import sample
import numpy as np
import torch.utils.data as data
from pathlib import Path
# Standard libraries
import torch.utils.data as data
import numpy as np
import pickle 
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
from os.path import join

def open_pickled_file(fn):
  with open(fn, "rb") as f_in:
    arr_new = pickle.load(f_in)
  return arr_new

from utils import open_image_np, open_target_get_class, open_target_get_class_with_perc 

class GeoFolders_2(data.Dataset):
    def get_match(self,i): 
        
        return [[x,i,i] for x in list(self.raw_dir.glob('*_Area_*'))if str(i.stem.split('_')[1]) in x.stem if i.stem.split("_")[3] in x.stem.split("_")[2] ][0] # long ikr!


    
    def __init__(self,root,transform,raw_dir,balance=True,k_labels_path=None):
        self.transform = transform
        self.raw_dir = raw_dir 
        self.raw_dir = Path(self.raw_dir)
        super(GeoFolders_2,self).__init__()
        self.k_labels_path = k_labels_path if k_labels_path is not None else None 
        list_regions = list(Path(root).glob("*_Area_*"))
        
        self.samples = []
        for i in list_regions:
            # print(i.stem.split('_')[1])
            self.samples.append(self.get_match(i))

        # pre-processing moved to init
        print("preprocessing paths ...")
        self.raw_paths = [sorted(list((raw_path[0]  /  "0").glob("*.pickle"))) for raw_path in self.samples]
        self.slice_paths = [sorted(list((slice_path[1]  /  "0").glob("*.png"))) for slice_path in self.samples]

        del self.samples
        print("Current smaple count ", len(self.raw_paths)* 1936)
    def __len__(self):
        return 1936 * len(self.raw_paths)  + 1

    def __getitem__(self, index: int):
        # which bin/region
        where = math.ceil(index / 1936) - 1 
        # which patch in that region 
        which = abs(((where) * 1936) - index ) - 1
        # maximum of each bin is 1936
        assert which <=1936
        # print(index,where,which)

        #get string paths - takes long - we search in each dir on each index lookup - very slow  
        # raw_path, slice,target = self.samples[where]
        raw_path = self.raw_paths[where][which]
        slice = self.slice_paths[where][which]
        target= slice
        target = str(target).replace("/64/","/64/anno/")
        



        #return actula color tile fro testing 
        color_path = open_image_np(slice)
        target= open_target_get_class_with_perc(target,0.5)
        input  = open_pickled_file(raw_path)
        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)
            # color_path=self.transform(color_path)

       
        return input,target,color_path
    def get_labels(self):
        
        return np.load(self.k_labels_path)

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



class GeoFolders(ImageFolder):

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
        color_path = np.array(Image.open(self.samples[index][0]))

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







class Whole_Slide_Bag(data.Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        # self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
        
        return img


class svs_h5_dataset(data.Dataset):
    def find_bin(self,y):
        l = [0]
        for ll,x in zip(l,self.tot):
            l.append(x+ll)
            where = list(map((lambda x: x-y ),l))
            which = bisect_right(where,0)
        return which
    def __init__(self, root_dir, split="all", transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(svs_h5_dataset,self).__init__()
        #train dir 
        img_dir = root_dir

        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if ".h5" in x ])

        # get total patches in each WSI
        tot=[]
        for can in range(len(self.image_filenames)):
            fn = self.image_filenames[can]
            tot.append(len(Whole_Slide_Bag(fn)))
        self.tot = tot
        
        self.target_filenames = []
        sp= self.image_filenames.__len__()
        sp= int(train_pct *sp)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split =='all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        #assert len(self.image_filenames) == len(self.target_filenames)
        tot=[]
        for can in range(len(self.image_filenames)):
            fn = self.image_filenames[can]
            tot.append(len(Whole_Slide_Bag(fn)))
        self.tot = tot
        # report the number of images in the dataset
        print('Number of {0} images: {1} svs'.format(split, self.__len__()))

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
        target = 0 #self.target_filenames[index]
        # get which Image 
        
        where = self.find_bin(index) - 1
        try:
            input = Whole_Slide_Bag(self.image_filenames[where])
        except:
            print(f"Couldnt find Image with index {where} with input index of {index}")

        # Which index in that image         
        which = index & len(input) - 1


        # load the nifti images
        if not self.preload_data:
            try:
                input = input[which]
            except:
                print(f"Couldn't find patch with index {which} in SVS with total of {len(input)} patches")
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self):
        return sum(self.tot)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array