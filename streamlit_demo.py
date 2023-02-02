from deeplab import deeplab
from helpers import create_stitched_image
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import streamlit as st
import os 
from PIL import Image
from torchvision import transforms
import torch
import pywick.transforms.tensor_transforms as ts
from datasets import GeoFolders_2_VALID, glas_dataset,GeoFolders,GeoFolders_2
from torch.utils.tensorboard._utils import make_grid

from pathlib import Path


raw_dir = '/home/uz1/data/geo/slices_raw/test/'
DATASET_PATH ="/home/uz1/data/geo/slices/test/"
regions = Path(DATASET_PATH).glob("*_Area_*")
regions = [ x for x in regions if "USA_Area_3" not in str(x) and "Egypt" not in str(x) and "China" not in str(x)]
regions = [x.parts[-1] for x in regions]
Full_image_PATH = "/home/uz1/data/geo/full_image/new_data/out/"
def get_transfroms():
    return transforms.Compose([
        # Resize((128, 128)),
        transforms.ToTensor(),
        ts.TypeCast(['float', 'float']),
        ts.StdNormalize(),
        ])
@st.cache(allow_output_mutation=True)
def load_dataset(p):
    
    transform_valid = get_transfroms()
    valid_dataset = GeoFolders_2_VALID(
            root=DATASET_PATH,  transform=transform_valid,raw_dir=raw_dir,split="valid",pick=p)
    input_imgs= [b for b in valid_dataset]
    input_img = torch.stack([x[0] for x in input_imgs],dim=0)# Images to reconstruct during training
    targets = torch.stack([torch.tensor(x[1]) for x in input_imgs])
    target_imgs = [x[2] for x in input_imgs]
    return input_img,targets,target_imgs
@st.cache(suppress_st_warning=True)
def load_image(p=2):
    raw_mdoel_reg = list((Path(DATASET_PATH).glob(f"*_Area_*")))
    raw_mdoel_reg = [x for x in raw_mdoel_reg if "USA_Area_3" not in str(x) and "Egypt" not in str(x) and "China" not in str(x)]
    raw_mdoel_reg = raw_mdoel_reg[p]
    raw_mdoel_reg = raw_mdoel_reg.parts[-1]
    model_region = list(Path(Full_image_PATH).glob("*_Area_*")) 
    model_region = [ x for x in model_region if "USA_Area_3" not in str(x) and "Egypt" not in str(x) and "China" not in str(x)]
    # print(model_region)
    if type(p) == int:
        #find raw_mdoel_reg in model_region
        image_path = [x for x in model_region if raw_mdoel_reg in str(x)]
        image_path = image_path[0]
        print("image path",image_path)
    else:
        image_path =[x for x in model_region if p in str(x)]
        image_path = image_path[0]
    image = Image.open(image_path)
    #display picked image 
    st.image(image)
def load_model(which_fold=0,p=2):
    model_path = str(sorted(list(list(Path('/home/uz1/saved_models_3dAE_NEW_DATA/').glob(f'./*{which_fold}__{p}.*'))[-1].glob('./*/*/*/*')))[-1])
    # extract region name for logging path of sprite image 
    model = deeplab(num_classes=9,proj_output_dim=1024,pred_hidden_dim=512,num_ch=9,num_predicted_clases=2)
    model = deeplab.load_from_checkpoint(model_path)
    return model



def predict(model,dataset,thresh):

    # print(dataset)
    x, y, color = dataset
    # y=torch.tensor(y)
    # x = x.unsqueeze(1)
    with torch.no_grad():
        x_hat, class_feat, class_logits = model(x)
    pred = model.apply_argmax_softmax(class_logits).argmax(1)
    #get probability of each class
    prob = torch.nn.functional.softmax(class_logits, dim=1)
    # st.write("prob ",prob)
    
    st.write("Thresh Count ->",(prob[:,1] > thresh).unique(return_counts=True)[1].numpy())
    # threshold the probability to get classfication
    pred = (prob[:,1] > thresh).float()
    # print(y.shape,color)
    f1 = f1_score(y.cpu(), pred.cpu(), average=None)
    precision = precision_score(y.cpu(), pred.cpu(), average=None)
    recall = recall_score(y.cpu(), pred.cpu(), average=None)

    
    fig = create_stitched_image(np.array(color),pred.int(),None)
    im,f1,precision,recall = Image.fromarray(np.uint8(make_grid(np.array(fig),7).transpose(1,2,0))),f1,precision,recall
    st.image(im)
    st.write("f1 score",f1,"precision",precision,"recall",recall)

def main():
    st.title('AI Model for Remote Sensing \n Hydrothemral Alteration detection')
    # p = st.selectbox("Select a region from below:",regions)

    # # find what index is p in regions
    # p_idx = regions.index(p)
    # print(p_idx)
    # image = load_image(p_idx)

    # dataset =  load_dataset(p)
    # st.write("Dataset loaded . . .")
    # model = load_model(p=p_idx)
    # st.write("Model loaded . . .")
    # thresh = st.slider('select threshold ', 0.0, 1.0, 0.0, 0.01)
    # result = st.button('Run on image')
    # st.write(st.session_state)
    # if result:
    #     st.write('Calculating results...')
    #     im,f1,precision,recall = predict(model, dataset,thresh)
    #     st.image(im)
    #     st.write("f1 score",f1)
    #     st.write("precision",precision)
    #     st.write("recall",recall)


    p = st.selectbox("Select a region from below:",regions)
    # find what index is p in regions
    p_idx = regions.index(p)
    print(p_idx)
    image = load_image(p_idx)
    dataset =  load_dataset(p)
    st.write("Dataset loaded . . .")
    model = load_model(p=p_idx)
    st.write("Model loaded . . .")
    thresh = st.slider('select threshold ', 0.0, 1.0, 0.5, 0.01)
    predict(model,dataset,thresh)


if __name__ == '__main__':
    main()