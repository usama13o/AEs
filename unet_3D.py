import math
from torch import optim
import torch
import torch.nn as nn
from util import HookBasedFeatureExtractor, SeqModelFeatureExtractor, UnetConv3, UnetUp3
import torch.nn.functional as F
from networks_other import init_weights
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class unet_3D(pl.LightningModule):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True,proj_output_dim=None,pred_hidden_dim =None):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.temperature = 0.004

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm,kernel_size=(1,3,3),padding_size=(0,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv,is_batchnorm = is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
         # projector
        self.projector = nn.Sequential(
            nn.Linear(filters[4] * 9* 4*4, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, 2),
        )
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs) #  9 64
        maxpool1 = self.maxpool1(conv1) # 9 32 

        conv2 = self.conv2(maxpool1) # 9 32
        maxpool2 = self.maxpool2(conv2) # 9 16

        conv3 = self.conv3(maxpool2) # 9 16
        maxpool3 = self.maxpool3(conv3)# 9 8 

        conv4 = self.conv4(maxpool3) # 9 8
        maxpool4 = self.maxpool4(conv4) # 9 4


        center = self.center(maxpool4)
        class_feat = self.projector(torch.flatten(center,1))
        class_logits = self.predictor(class_feat)
        up4 = self.up_concat4(conv4, center) # 9 12 
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final,class_feat , class_logits
    def get_feature_maps(self,x,layers):
        x,y,_  = x
        x=x.unsqueeze(1)
        feature_extractor = SeqModelFeatureExtractor(self,'all')
        inputs= feature_extractor.forward(x)


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x,y,_ = batch
        x = x.unsqueeze(1)
        x_hat,class_feat,class_logits = self.forward(x)
        class_loss = F.cross_entropy(self.apply_argmax_softmax(class_logits),y,weight=torch.tensor([0.6,1.5]).to(self.device))
        nn_loss = self.nn_loss(class_feat,y)
        pred = self.apply_argmax_softmax(class_logits).argmax(1)
        f1 = f1_score(y.cpu(),pred.cpu(),average='macro')

        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3,4]).mean(dim=[0])
        return (loss , class_loss, nn_loss),f1
    def nn_loss(self,pred,labels):
        """
        pred: (N,C) prediction from projection layer 
        labves: (N) labels for each of the datapoints. Th ground truth values 
        """

        # pos_quueu = [pred[i] for i,x in enumerate(labels) if x==1]
        # neg_quueu = [pred[i] for i,x in enumerate(labels) if x==0]
        y = labels.view(-1,1)
        pred = F.normalize(pred,dim=1)

        mask = torch.eq(y,y.T).float()

        contrast_count = pred.shape[0]
        contrast_feature = pred
        anchor_feature= pred
        anchor_count = pred.shape[0]

        # impose dissamilarity 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(anchor_count ).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask[:logits.shape[0],:]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask[:logits.shape[0],:] * log_prob).sum(1) / mask[:logits.shape[0],:].sum(1)

        # loss
        loss = - (self.temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count).mean()
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss,f1= self._get_reconstruction_loss(batch)
        self.log_dict({'f1_train':f1,'recon_loss': loss[0],'class_loss':loss[1],'nn_loss':loss[2]})
        return loss[1] 

    def validation_step(self, batch, batch_idx):
        # self.get_feature_maps(batch,'conv3')
        loss,f1 = self._get_reconstruction_loss(batch)
        self.log_dict({'f1_valid':f1,'val_loss': loss[0] + loss[1],'class_loss_val':loss[1],"nn_loss_val":loss[2]})

    def test_step(self, batch, batch_idx):
        loss,f1 = self._get_reconstruction_loss(batch)
        self.log_dict({"f1_test":f1,'test_recon_loss': loss[0],'class_loss_test':loss[1]})













