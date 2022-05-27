from numpy import ndarray
from pywick.models.segmentation.deeplab_v3 import DeepLabv3
import ttach as tta
from pywick.models.segmentation.deeplab_v3_plus import Atrous_Bottleneck, Atrous_ResNet_features, Atrous_module, DeepLabv3_plus
from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score


class deeplab(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 small=True,
                 pretrained=True,
                 proj_output_dim=None,
                 pred_hidden_dim=None,
                 num_predicted_clases=2,
                 **kwargs):
        # self.num_classes= num_classes
        super().__init__()
        block = Atrous_Bottleneck
        self.temperature = 0.004
        self.save_hyperparameters()

        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23],
                                                      kwargs['num_ch'],
                                                      pretrained)
        self.resnet_features.conv1 = self.conv1 = nn.Conv2d(kwargs['num_ch'],
                                                            64,
                                                            kernel_size=7,
                                                            stride=2,
                                                            padding=3,
                                                            bias=False)
        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(2048, 256, kernel_size=1))

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),
                                 nn.BatchNorm2d(256))

        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
                                          nn.BatchNorm2d(48))
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.feature_scale = 4
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.projector = nn.Sequential(
            nn.Linear(2048 * 4 * 4, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, num_predicted_clases),
        )

    def forward(self, x):
        x, conv2 = self.resnet_features(x)
        class_feat = self.projector(torch.flatten(x, 1))
        class_logits = self.predictor(class_feat)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear')

        low_lebel_features = self.reduce_conv2(conv2)

        x = torch.cat((x, low_lebel_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear')

        return x, class_feat, class_logits

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

    def _get_reconstruction_loss(self, batch):
        """
                Given a batch of images, this function returns the reconstruction loss (MSE in our case)
                """
        x, y, color = batch
        # x = x.unsqueeze(1)
        x_hat, class_feat, class_logits = self.forward(x)
        class_loss = F.cross_entropy(self.apply_argmax_softmax(class_logits),
                                     y)
        nn_loss = self.nn_loss(class_feat, y)
        pred = self.apply_argmax_softmax(class_logits).argmax(1)
        f1 = f1_score(y.cpu(), pred.cpu(), average=None)
        precision = precision_score(y.cpu(), pred.cpu(), average=None)
        recall = recall_score(y.cpu(), pred.cpu(), average=None)
        noc=[len(y[y==1]),recall[1] * len(y[y==1])] if 1 in y else [0,0] # number of correctly classified for ROI 
        if color.shape[0] > 5:
            color = color.permute(0, 3, 1, 2)
            loss = F.mse_loss(x, x_hat, reduction="none")
        else:
            loss = F.mse_loss(color, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return (loss, class_loss, nn_loss), f1, precision, recall,noc

    def nn_loss(self, pred, labels):
        """
                pred: (N,C) prediction from projection layer 
                labves: (N) labels for each of the datapoints. Th ground truth values 
                """

        # pos_quueu = [pred[i] for i,x in enumerate(labels) if x==1]
        # neg_quueu = [pred[i] for i,x in enumerate(labels) if x==0]
        y = labels.view(-1, 1)
        pred = F.normalize(pred, dim=1)

        mask = torch.eq(y, y.T).float()

        contrast_count = pred.shape[0]
        contrast_feature = pred
        anchor_feature = pred
        anchor_count = pred.shape[0]

        # impose dissamilarity
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask[:logits.shape[0], :]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask[:logits.shape[0], :] * log_prob
                             ).sum(1) / mask[:logits.shape[0], :].sum(1)

        # loss
        loss = -(self.temperature) * mean_log_prob_pos
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "class_loss_val"
        }

    def training_step(self, batch, batch_idx):
        loss, f1, precision, recall,_ = self._get_reconstruction_loss(batch)
        self.log_dict({
            'f1_0_train': f1[0],
            'f1_1_train': f1[1],
            "precision_0_train": precision[0],
            "precision_1_train": precision[1],
            "recall_0_train":recall[0],
            "recall_1_train":recall[1],
            'recon_loss': loss[0],
            'class_loss': loss[1],
            'nn_loss': loss[2]
        })
        return loss[1]

    def validation_step(self, batch, batch_idx):
        # self.get_feature_maps(batch,'conv3')
        loss, f1, precision, recall ,_= self._get_reconstruction_loss(batch)
        self.log_dict({
            'f1_0_val': f1[0],
            'f1_1_val': f1[1] if len(f1) > 1 else 0,
            "precision_0_val": precision[0] ,
            "precision_1_val": precision[1] if len(precision) > 1 else 0,
            "recall_0_val":recall[0],
            "recall_1_val":recall[1] if len(recall) > 1 else 0,
            'val_loss': loss[0] + loss[1],
            'class_loss_val': loss[1],
            "nn_loss_val": loss[2]
        })

    def test_step(self, batch, batch_idx):
        loss, f1, precision, recall,noc = self._get_reconstruction_loss(batch)
        x,y,_= batch
        # model =  tta.ClassificationTTAWrapper(self,tta.aliases.d4_transform())
        # res = model(x,y.cpu())
        # res = res[0] if isinstance(res,ndarray) else res
        # res=(res > 1).int()
        self.log_dict({
            'f1_0_test': f1[0],
            'f1_1_test': f1[1] if len(f1) > 1 else 0,
            "precision_0_test": precision[0],
            "precision_1_test": precision[1] if len(precision) > 1 else 0,
            "recall_0_test":recall[0],
            "recall_1_test":recall[1] if len(recall) > 1 else 0,
            'test_recon_loss': loss[0],
            'class_loss_test': loss[1],
            'Number of Correctly Classified':noc[1],
            "Number of ROI regions":noc[0]
            # "mean_f1_0_tta": res,
        })
