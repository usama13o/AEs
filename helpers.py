
import math
## PyTorch
from typing import Iterable
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np 
from torch.utils.tensorboard._embedding import make_sprite
from PIL import ImageDraw, ImageFont

# Cell
class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m, hook_func, is_forward=True, detach=True, cpu=False, gather=False):
        self.hook_func = hook_func 
        self.detach= detach
        self.cpu = cpu
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.stored,self.removed = None,False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input,output = input[0].detach() , output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

    _docs = dict(__enter__="Register the hook",
                 __exit__="Remove the hook")

# Cell
def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else list(o)

def hook_output(module, detach=True, cpu=False, grad=False):
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)

# Cell

class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms, hook_func, is_forward=True, detach=True, cpu=False):
        self.hooks = [Hook(m, hook_func, is_forward, detach, cpu) for m in ms]

    def __getitem__(self,i): return self.hooks[i]
    def __len__(self):       return len(self.hooks)
    def __iter__(self):      return iter(self.hooks)
    @property
    def stored(self):        return list(o.stored for o in self)

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()

    _docs = dict(stored = "The states saved in each hook.",
                 __enter__="Register the hooks",
                 __exit__="Remove the hooks")
def listify(p, q):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

def splitAtFirstParenthesis(s,showDetails,shapeData):
    pos=len(s.split('(')[0])
    ret = s[:pos]
    if (showDetails):
        ret += shapeData + '\n' + s[pos:]
    return ret
class HookBasedFeatureExractorCallback(pl.Callback):
    def getHistImg(self,act,useClasses):
        dd = act.squeeze(2) if not useClasses else act.sum(dim=2) # Reassemble...
        dd = dd.log() # Scale for visualizaion
        dd = dd.t() # rotate
        return dd
    def mkHist(self, x, useClasses):
        ret = x.clone().detach().cpu() # Need this for Pytorch 1.0
        #ret = x.clone().detach() #Pytorch 1.1 # WARNING: so bad performance on GPU! (x10)
        if useClasses:
            ret = torch.stack([ret[:,i].histc(self.nBins, self.hMin, self.hMax) for i in range(ret.shape[1])],dim=1) #histogram by class...
        else:
            ret = ret.histc(self.nBins, self.hMin, self.hMax).unsqueeze(1) # histogram by activation
        return ret
    def __init__(self,layers=None,hMin=-20,
                hMax=20,
                nBins=500,) -> None:
        super().__init__()
        self.layers=layers
        self.hMin = hMin or (-hMax)
        self.hMax = hMax
        self.nBins = nBins
        self.stats_hist = None # empty at start
        self.stats_valid_hist = None # empty at start
        self.stats_epoch = []
        self.cur_epoch = -1
        self.cur_train_batch = -1
        self.stats_valid_epoch = []
        self.shape_out={}
        self.useClasses = False

    def hook(self, m:nn.Module, i, o):
        if (isinstance(o,torch.Tensor)) and (m not in self.shape_out):
            outShape = o.shape; 
            self.shape_out[m]=outShape;
        return self.mkHist(o,False)
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.modules = [m[1] for m in pl_module._modules.items()]
        self.act_means = [[] for _ in self.modules]
        self.act_stds  = [[] for _ in self.modules]
        self.act_hist  = [[] for _ in self.modules]
        self.hooks = Hooks(self.modules, self.hook, True, True , True)
    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch:
            self.cur_train_batch +=1
    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        "Take the stored results and puts it in `self.stats_hist`"
        hasValues = True if ((len(self.hooks.stored)>0) and (not (self.hooks.stored[0] is None))) else False
        stacked = torch.stack(self.hooks.stored).unsqueeze(1)  if hasValues else None
        # if train and hasValues:
        if self.stats_hist is None: self.stats_hist = stacked #start
        else: self.stats_hist = torch.cat([self.stats_hist,stacked],dim=1) #cat
    #     if (not train) and hasValues:
    #         if self.stats_valid_hist is None: self.stats_valid_hist = stacked #start
    #         else: self.stats_valid_hist = torch.cat([self.stats_valid_hist,stacked],dim=1) #cat
    # def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # if trainer.current_epoch % 10:
        self.plotActsHist()
    
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.stats_epoch.append(self.cur_train_batch)
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._remove()
    def _remove(self):
        if getattr(self, 'hooks', None): self.hooks.remove()
    def __del__(self): self._remove()

    def plotActsHist(self, cols=3, figsize=(50,20), toDisplay=None, hScale = .05, showEpochs=True, showLayerInfo=False, aspectAuto=True, showImage=True):
            histsTensor = self.stats_hist
            hists = [histsTensor[i] for i in range(histsTensor.shape[0])]
            if toDisplay: hists = [hists[i] for i in listify(toDisplay)] # optionally focus

            n=len(hists)
            cols = cols or 3
            cols = min(cols,n)
            rows = int(math.ceil(n/cols))
            fig = plt.figure(figsize=figsize)
            grid = plt.GridSpec(rows, cols, figure=fig)

            for i,l in enumerate(hists):
                img=self.getHistImg(l,self.useClasses)
                cr = math.floor(i/cols)
                cc = i%cols
                main_ax = fig.add_subplot(grid[cr,cc])
                if showImage: main_ax.imshow(img); 
                layerId = listify(toDisplay)[i] if toDisplay else i 
                m = self.modules[layerId]
                outShapeText = f'  (out: {list(self.shape_out[m])})' if (m in self.shape_out) else ''
                title = f'L:{layerId}' + '\n' + splitAtFirstParenthesis(str(m),showLayerInfo,outShapeText)
                main_ax.set_title(title)
                imgH=img.shape[0]
                main_ax.set_yticks([])
                main_ax.set_ylabel(str(self.hMin) + " : " + str(self.hMax))
                if aspectAuto: main_ax.set_aspect('auto')
                imgW=img.shape[1]
                imgH=img.shape[0]
                ratioH=-self.hMin/(self.hMax-self.hMin)
                zeroPosH = imgH*ratioH
                main_ax.plot([0,imgW],[zeroPosH,zeroPosH],'r') # X Axis
                if (showEpochs):
                    start = 0
                    nEpochs = len(self.stats_epoch)
                    for i,hh in enumerate(self.stats_epoch):
                        if(i<(nEpochs-1)): main_ax.plot([hh,hh],[0,imgH],color=[0,0,1])
                        end = hh # rolling
                        domain = l[start:end]
                        domain_mean = domain.mean(-1) # mean on classes
                        if self.useClasses:
                            plotPerc(main_ax,domain,hScale,1,start,colorById=True,addLabel=(0==i)) #plot all
                            main_ax.legend(loc='upper left')
                        else:
                            plotPerc(main_ax,domain_mean,hScale,.5,start)
                        plotPerc(main_ax,domain_mean,hScale,1,start,linewidth=1.5)
                        start = hh
                main_ax.set_xlim([0,imgW])
                main_ax.set_ylim([0,imgH])
            fig.savefig('test_hist_!.png')
            return fig

def computeXY(l,hscale,perc,hshift=0):
        start = int(l.shape[0]*(1-perc))
        end = l.shape[0]
        m = l[start:end].mean(dim=0) # all data mean
        xx = hshift + m*hscale
        yy = +np.array(range(l.shape[1]))
        return xx,yy
def get_color_value_from_map(idx:int, cmap='Reds', scale=1):
    return plt.get_cmap(cmap)(idx*scale)

def plotPerc(ax,l,hscale,perc,hshift=0,colorById=False,linewidth=1,addLabel=False):
    xx,yy = computeXY(l,hscale,perc,hshift)
    if colorById:
        classes = xx.shape[1] 
        for i in range(classes):
            xx_cur = xx[:,i]
            color = get_color_value_from_map(i/classes, cmap='rainbow')
            label = i if addLabel else None
            ax.plot(xx_cur,yy,linewidth=linewidth, color=color, label=label);
    else:
        color = [1-perc,1-perc,1-perc]
        ax.plot(xx,yy,linewidth=linewidth, color=color);
class GenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = torch.stack([x[0] for x in input_imgs],dim=0)# Images to reconstruct during training
        self.targets = [x[1] for x in input_imgs]
        self.target_imgs = [x[2] for x in input_imgs]
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
    def plot_imgs(self,pred,n_cols=4):
        figure = plt.figure(figsize=(12,8))
        n_rows=int(len(pred) /n_cols)

        for i in range(len(self.targets)):    
            plt.subplot(n_rows, n_cols, i + 1)
            plt.xlabel(f"GT: {self.targets[i]}  predicted : {pred[i]}")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.target_imgs[i], cmap=plt.cm.coolwarm)
        return figure
        
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device).unsqueeze(1)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs,_,pred = pl_module(input_imgs)
                pl_module.train()
            #log prediction for classification 
            pred = pred.softmax(1).argmax(1)
            fig = self.plot_imgs(pred)
            trainer.logger.experiment.add_figure('Predictions',fig,global_step=trainer.global_step)
            # Plot and add to tensorboard
            imgs = torch.stack([*input_imgs[1].squeeze(),*reconst_imgs[1].squeeze()],dim=0).unsqueeze(1)
            # imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            try:
                trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
            except:
                trainer.logger.experiment.log({"Reconstructions": grid})
class GenerateTestCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = torch.stack([x[0] for x in input_imgs],dim=0)# Images to reconstruct during training
        self.targets = [x[1] for x in input_imgs]
        self.target_imgs = [x[2] for x in input_imgs]
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
    def plot_imgs(self,pred,n_cols=4):
        figure = plt.figure(figsize=(12,8))
        n_rows=int(len(pred) /n_cols)

        for i in range(len(self.targets)):    
            plt.subplot(n_rows, n_cols, i + 1)
            plt.xlabel(f"GT: {self.targets[i]}  predicted : {pred[i]}")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.target_imgs[i], cmap=plt.cm.coolwarm)
        return figure
        
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device).unsqueeze(1)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs,embeds,pred = pl_module(input_imgs)
                pl_module.train()
            #log prediction for classification 
            pred = pred.softmax(1).argmax(1)
            fig = create_stitched_image(np.array(self.target_imgs),pred)
            
            # trainer.logger.experiment.add_figure('Predictions',fig,global_step=trainer.global_step)
            
            # Plot and add to tensorboard
            imgs = torch.stack([*input_imgs[1].squeeze(),*reconst_imgs[1].squeeze()],dim=0).unsqueeze(1)
            trainer.logger.experiment.add_embedding(embeds,  # Encodings per image
                     label_img=fig[:,:3,:,:], global_step=trainer.global_step)
            # imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            try:
                trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
            except:
                trainer.logger.experiment.log({"Reconstructions": grid})


def create_stitched_image(images,labels):
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
        im = im
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
    make_sprite(mod_images[:,:3,:,:], save_path='./')
    return mod_images[:,:,:,:]