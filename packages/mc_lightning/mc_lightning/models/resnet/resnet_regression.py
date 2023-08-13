import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models
from argparse import ArgumentParser
import numpy as np

class PretrainedResnet50FT_reg(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*self.image_modules)
        self.flattenn = torch.flatten
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        # out = self.image_modules[0](x)
        # for image_module in self.image_modules[1:]:
        #     out = image_module(out)
        out = self.resnet(x)
        # print('in forward, after resnet')
        # print(out.size())
        out = torch.flatten(out, 1)
        # print('in forward, after flatten')
        # print(out.size())
        # out = self.scaler(out)
        # print('in forward, after scaler')
        # print(out.size())
        out = self.classifier(out)
        # print('in forward, after classifier')
        # print(out.size()) 
        return out

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, label, slide_id = batch
        preds = self(x)        
        loss = torch.nn.functional.smooth_l1_loss(preds, label.resize(len(label), 1)) #Todo Document this

        tensorboard_logs = {'train_loss': loss}
        #         self.logger.experiment.add_scalar('train_loss', loss)
            
        # self.log('train_acc_step', train_acc)

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, label, slide_id = batch
        preds = self(x)        
        loss = torch.nn.functional.smooth_l1_loss(preds, label.resize(len(label), 1)) #Todo Document this
        batch_results = {
            'val_loss': loss,
            'val_preds': preds,
            'val_label': label,
            'val_slide_id': slide_id
        }

        # self.log('val_acc', val_acc)
        return batch_results

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # aggregate other results to pass to metric calculation
        all_preds = torch.cat([x['val_preds'] for x in outputs])
        all_labels = torch.cat([x['val_label'] for x in outputs])
        all_slide_ids = np.concatenate([x['val_slide_id'] for x in outputs])
       
        avg_pred = all_preds.float().mean()
        avg_label = all_labels.float().mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_avg_pred': avg_pred, 'val_avg_label': avg_label}
                            
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, label, slide_id = batch
        logits = self(x)
        loss = F.cross_entropy(logits, label)
        return {'test_loss': F.cross_entropy(logits, label)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def groupby_agg_mean(self, metric, labels):
        """
        https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2
        """
        labels = labels.unsqueeze(1).expand(-1, metric.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        #res = torch.zeros_like(unique_labels, dtype=metric.dtype).scatter_add_(0, labels, metric)
        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, metric)
        res = res / labels_count.float().unsqueeze(1)

        return res
