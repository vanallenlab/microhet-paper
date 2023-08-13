
"""
TODO. 
1. Model that optimizes both for abstention and number of examples answered 
"""


import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
import wandb 
import math
from torchmetrics import SpearmanCorrcoef
import torchmetrics

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class PretrainedResnet50FT(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--dropout', type=float, default=0.2)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1)
        out = self.dropout(out)     
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch
        
        #Av labels
        self.log(who + '_av_label', torch.mean(task_labels.float()))

        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "mean")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_loss', loss)
        self.log(who + '_acc', task_acc)
        self.log(who + '_f1', task_f1)

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)

        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

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

class PretrainedResnet50FTbecoremb(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_sources', type=int, default=33)
        parser.add_argument('--srcs_map', type=dict)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')  
              
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet1 = nn.Sequential(*image_modules)
        # self.resnet2 = nn.Sequential(*image_modules)
        self.emb_2_src = nn.Linear(2048, 2048)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.src_classifier = nn.Linear(2048, self.hparams.num_srcs)
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings

    def forward(self, x):
        task_embs = torch.flatten(self.resnet1(x), 1)        
        return task_embs

    def cos_sim(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        #Get the source from the slide id
        src_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id]).to('cuda')

        #Get the embeddings for the task and source
        task_embs = self(x)
        src_embeddings = F.relu(self.emb_2_src(task_embs))

        # self.log(who + '_src_embeddings_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_task_embeddings_norm', torch.mean(LA.norm(task_embs, dim=1))) 

        #Define some combination of task and source embeddings to be the final embeddings
        embeddings = self.combine_embeddings(task_embs, src_embeddings) 
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embeddings)
        src_logits = self.src_classifier(src_embeddings)

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        src_loss = F.cross_entropy(src_logits, src_labels, reduction = "none") 
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        cos_sim = cos(embeddings, src_embeddings) #torch.mean(torch.abs()) 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(src_loss, task_loss, cos_sim)
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Source acc
        src_preds = src_logits.argmax(-1)
        src_acc = torchmetrics.functional.accuracy(src_preds, src_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        src_f1 =  torchmetrics.functional.f1(src_preds, src_labels, num_classes = self.hparams.num_srcs, average = 'weighted')

        self.log(who + '_task_f1', task_f1)
        self.log(who + '_src_f1', src_f1)

        self.log(who + '_cos_sim', torch.mean(cos_sim))
        self.log(who + '_abs_cos_sim', torch.mean(torch.abs(cos_sim)))
        self.log(who + '_loss', loss)
        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_src_loss', torch.mean(src_loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_acc', task_acc)
        self.log(who + '_src_acc', src_acc)

        return loss

   
    
    def training_step(self, batch, batch_nb):
        # REQUIRED        
        loss = self.cos_sim('train', batch, batch_nb)

        return loss


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        
        loss = self.cos_sim('val', batch, batch_nb) 

        # give actual int preds; not sure if they can handle logits as above
        all_preds = all_logits.argmax(-1)
        val_auroc = pl.metrics.functional.auroc(all_preds, all_labels)
        val_acc = pl.metrics.functional.accuracy(all_preds, all_labels)
        val_f1 = pl.metrics.functional.f1_score(all_preds, all_labels)

        tensorboard_logs = {'val_loss': avg_loss, 'val_auroc': val_auroc, 'val_acc': val_acc, 'val_f1': val_f1, 'val_avg_pred': avg_pred, 'val_avg_label': avg_label}

        return loss

    


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        loss = self.cos_sim('test', batch, batch_nb) 
        return {'test_loss':loss}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss}

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

def cossim(output, target):
    loss = torch.mean((output - target)**2)
    return loss

class PretrainedResnet50FT_contrastive_multitask(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--srcs_map', type=dict)
        parser.add_argument('--include_ce_loss', type=bool)

        if self.hparams.include_ce_loss: 
            print('loss includes cosine loss and src loss')           
        else:
            print('loss includes only cosine loss')           
        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.embedder = nn.Linear(2048, 1024)
        self.classifier = nn.Linear(1024, self.hparams.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.resnet(x)
        out = self.relu(torch.flatten(out, 1))
        out = self.relu(self.embedder(out))
        return out
    
    def bio_src_site_step(self, who, batch):
        x1, x2, label, slide_id1, slide_id2  = batch

        #Get the source from the slide id
        src1_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id1]).to('cuda')
        src2_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id2]).to('cuda')
                
        embs1 = self(x1)        
        embs2 = self(x2)

        #Contrastive loss
        criterion = nn.CosineEmbeddingLoss()
        cosine_emd_loss = criterion(embs1, embs2, (2*label-1).float())
        src1_logits = self.classifier(embs1)
        src2_logits = self.classifier(embs2)
        
        #Src Pred Train acc
        src1_preds = src1_logits.argmax(-1)
        src1_loss = F.cross_entropy(src1_logits, src1_labels) #Surya's change        

        src2_preds = src1_logits.argmax(-1)
        src2_loss = F.cross_entropy(src2_logits, src2_labels) #Surya's change        

        src1_acc = torchmetrics.functional.accuracy(src1_preds, src1_labels) 
        src2_acc = torchmetrics.functional.accuracy(src2_preds, src2_labels)

        src1_f1 = torchmetrics.functional.f1(src1_preds, src1_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        src2_f1 = torchmetrics.functional.f1(src2_preds, src2_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        if self.hparams.include_ce_loss:            
            # print('loss includes cosine loss and src loss')           
            loss = src1_loss + cosine_emd_loss  #+ src2_loss
            # print(src_loss, cosine_emd_loss, loss)
        else:
            # print('loss includes only cosine loss')           
            loss = cosine_emd_loss
            # print(src_loss, cosine_emd_loss, loss)

        self.log(who + '_cosine_emd_loss', cosine_emd_loss)
        self.log(who + '_src_loss', src1_loss + src2_loss)
        self.log(who + '_loss', loss)
        self.log(who + '_src1_acc', src1_acc)
        self.log(who + '_src2_acc', src2_acc)
        self.log(who + '_src1_f1', src1_f1)
        self.log(who + '_src2_f1', src2_f1)
        

        return loss 

    def training_step(self, batch, batch_nb):
        loss = self.bio_src_site_step('train', batch)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.bio_src_site_step('val', batch)
        return loss

    def test_step(self, batch, batch_nb):
        loss = self.bio_src_site_step('test', batch)
        return loss

        return {'test_loss': avg_loss, 'log': tensorboard_logs}
    
    
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

class Bekind_mh(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_sources', type=int, default=33)
        parser.add_argument('--srcs_map', type=dict)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin') 
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        # self.resnet2 = nn.Sequential(*image_modules)
        self.emb_2_src = nn.Linear(2048, 2048)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.src_classifier = nn.Linear(2048, self.hparams.num_srcs)
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        self.model_type = self.hparams.model_type

        self.non_lin = self.hparams.non_lin

    def forward(self, x):

        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1)    )
        src_embeddings = self.non_lin(self.emb_2_src(img_embs))
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        #Define some combination of task and source embeddings to be the final embeddings
        task_embeddings = self.combine_embeddings(img_embs, src_embeddings) 

        return img_embs, src_embeddings, task_embeddings

    def cos_sim(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        #Get the source from the slide id
        src_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id]).to('cuda')
        
        #Define some combination of task and source embeddings to be the final embeddings
        img_embs, src_embeddings, task_embeddings = self(x)
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        src_logits = self.src_classifier(src_embeddings)

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        src_loss = F.cross_entropy(src_logits, src_labels, reduction = "none") 
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        cos_sim = cos(task_embeddings, src_embeddings) 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(src_loss, task_loss, cos_sim)
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Source acc
        src_preds = src_logits.argmax(-1)
        src_acc = torchmetrics.functional.accuracy(src_preds, src_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        src_f1 =  torchmetrics.functional.f1(src_preds, src_labels, num_classes = self.hparams.num_srcs, average = 'weighted')

        self.log(who + '_task_f1', task_f1)
        self.log(who + '_src_f1', src_f1)

        self.log(who + '_cos_sim', torch.mean(cos_sim))
        self.log(who + '_abs_cos_sim', torch.mean(torch.abs(cos_sim)))        
        self.log(who + '_loss', loss)        
        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_src_loss', torch.mean(src_loss))

        self.log(who + '_task_acc', task_acc)
        self.log(who + '_src_acc', src_acc)

        if who == 'train':
            
            if self.model_type == 'becor':
                task_opt, _ = self.optimizers()

                self.manual_backward(loss, task_opt)

                task_opt.step()

            elif self.model_type == 'normal':
                task_opt, src_opt = self.optimizers()

                self.manual_backward(loss, task_opt, retain_graph = True)
                self.manual_backward(torch.mean(src_loss), src_opt)

                task_opt.step()
                src_opt.step()

            

    def training_step(self, batch, batch_nb, optimizer_idx):
        # REQUIRED        
        loss = self.cos_sim('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        loss = self.cos_sim('val', batch, batch_nb) 
        return loss


    def test_step(self, batch, batch_nb):
        # OPTIONAL
        loss = self.cos_sim('test', batch, batch_nb) 
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        return 

    def configure_optimizers(self):
        
        if self.model_type == 'normal':
            task_params = list(self.resnet.parameters()) + list(self.classifier.parameters()) 
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr)
            
        elif self.model_type == 'becor':
            task_params = self.parameters()
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr)            
        
        src_params = list(self.src_classifier.parameters()) + list(self.emb_2_src.parameters()) 
        src_opt = torch.optim.Adam(src_params, lr=self.hparams.lr) 

        return [task_opt, src_opt]
        

        


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

class Bekind_sl(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_sources', type=int, default=33)
        parser.add_argument('--num_slides', type=int, default=150)
        parser.add_argument('--srcs_map', type=dict)        
        parser.add_argument('--slides_map', type=dict)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--lr', type=float, default=1e-5)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        # self.resnet2 = nn.Sequential(*image_modules)
        
        self.emb_2_src = nn.Linear(2048, 2048)
        self.emb_2_sl = nn.Linear(2048, 2048)
        
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.src_classifier = nn.Linear(2048, self.hparams.num_srcs)
        self.sl_classifier = nn.Linear(2048, self.hparams.num_slides)

        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.model_type = self.hparams.model_type

        self.non_lin = self.hparams.non_lin

    def forward(self, x):

        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        src_embeddings = self.emb_2_src(img_embs)
        slide_embeddings = self.emb_2_sl(img_embs)

        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        #Introduce non-linearity
        src_embeddings, slide_embeddings = (self.non_lin(x) for x in (src_embeddings, slide_embeddings))

        return img_embs, src_embeddings, slide_embeddings

    def cos_sim(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        self.log(who + '_av_label', sum(task_labels) / len(task_labels))
        
        img_embs, src_embeddings, slide_embeddings = self(x)
        
        #Define some combination of task and source embeddings to be the final embeddings
        task_embeddings = self.combine_embeddings(task_embeddings = img_embs, src_embeddings = src_embeddings,  slide_embeddings = slide_embeddings) 

        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        src_cos_sim = cos(task_embeddings, src_embeddings) 
        slide_cos_sim = cos(task_embeddings, slide_embeddings)
        slide_src_cos_sim = cos(src_embeddings, slide_embeddings)
        
        #Get the source from the slide id
        src_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id]).to('cuda')
        slide_labels = torch.LongTensor([self.hparams.slides_map[i] for i in slide_id]).to('cuda')
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        src_logits = self.src_classifier(src_embeddings)
        slide_logits = self.sl_classifier(slide_embeddings)

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        src_loss = F.cross_entropy(src_logits, src_labels, reduction = "none") 
        slide_loss = F.cross_entropy(slide_logits, slide_labels, reduction = "none") 
        
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(   src_loss = src_loss,
                                    slide_loss = slide_loss,
                                    task_loss= task_loss,
                                    src_cos_sim= src_cos_sim,
                                    slide_cos_sim = slide_cos_sim,
                                    slide_src_cos_sim = slide_src_cos_sim
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Source acc
        src_preds = src_logits.argmax(-1)
        src_acc = torchmetrics.functional.accuracy(src_preds, src_labels)

        #Slide acc
        slide_preds = slide_logits.argmax(-1)
        slide_acc = torchmetrics.functional.accuracy(slide_preds, slide_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        src_f1 =  torchmetrics.functional.f1(src_preds, src_labels, num_classes = self.hparams.num_srcs, average = 'weighted')
        slide_f1 =  torchmetrics.functional.f1(slide_preds, slide_labels, num_classes = self.hparams.num_slides, average = 'weighted')

        self.log(who + '_task_f1', task_f1)
        self.log(who + '_src_f1', src_f1)
        self.log(who + '_slide_f1', slide_f1)

        self.log(who + '_src_cos_sim', torch.mean(src_cos_sim))
        self.log(who + '_abs_src_cos_sim', torch.mean(torch.abs(src_cos_sim)))        
        self.log(who + '_slide_cos_sim', torch.mean(slide_cos_sim))
        self.log(who + '_abs_slide_cos_sim', torch.mean(torch.abs(slide_cos_sim))) 
        self.log(who + '_slide_src_cos_sim', torch.mean(slide_src_cos_sim))
        self.log(who + '_abs_slide_src_cos_sim', torch.mean(torch.abs(slide_src_cos_sim)))        
        
        self.log(who + '_loss', loss)        
        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_src_loss', torch.mean(src_loss))
        self.log(who + '_slide_loss', torch.mean(slide_loss))

        self.log(who + '_task_acc', task_acc)
        self.log(who + '_src_acc', src_acc)
        self.log(who + '_slide_acc', slide_acc)

        if who == 'train':
            
            if self.model_type == 'becor':
                task_opt, _, _ = self.optimizers()

                self.manual_backward(loss, task_opt)

                task_opt.step()

            elif self.model_type == 'normal':
                task_opt, src_opt, slide_opt = self.optimizers()

                self.manual_backward(loss, retain_graph = True)
                self.manual_backward(torch.mean(src_loss), retain_graph=True)
                self.manual_backward(torch.mean(slide_loss))
                
                task_opt.step()
                src_opt.step()
                slide_opt.step()

            

    def training_step(self, batch, batch_nb, optimizer_idx):
        # REQUIRED        
        loss = self.cos_sim('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        loss = self.cos_sim('val', batch, batch_nb) 
        return loss


    def test_step(self, batch, batch_nb):
        # OPTIONAL
        loss = self.cos_sim('test', batch, batch_nb) 
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        return 

    def configure_optimizers(self):
        
        if self.model_type == 'normal':
            task_params = list(self.resnet.parameters()) + list(self.classifier.parameters()) 
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr)
            
        elif self.model_type == 'becor':
            task_params = self.parameters()
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr)            
        
        src_params = list(self.resnet.parameters()) + list(self.src_classifier.parameters()) + list(self.emb_2_src.parameters()) 
        src_opt = torch.optim.Adam(src_params, lr=self.hparams.lr) 
        
        slide_params = list(self.resnet.parameters()) + list(self.sl_classifier.parameters()) + list(self.emb_2_sl.parameters()) 
        slide_opt = torch.optim.Adam(slide_params, lr=self.hparams.lr) 

        return [task_opt, src_opt, slide_opt]

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

class PretrainedResnet50FT_reg(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

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

class Bekind_str(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False
        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        if self.model_type == 'becor':
            self.classifier = nn.Linear(2048 * 2, self.hparams.num_classes)
        
        elif self.model_type == 'normal':
            self.classifier = nn.Linear(2048, self.hparams.num_classes)
        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=0.2)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_f1"] = 0

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        
        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)

        stain_embeddings = self.non_lin(self.emb_2_stain(img_embs))        
        task_embeddings = self.non_lin(self.emb_2_task(img_embs))

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        #Define some combination of task and source embeddings to be the final embeddings
        task_embeddings = self.combine_embeddings(
                                                    task_embeddings = task_embeddings,
                                                    stain_embeddings = stain_embeddings
                                                ) 

        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Stain acc
        stain_preds = stain_logits.argmax(-1)
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'weighted')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_f1"], task_f1)

        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        if who == 'train':

            if self.model_type == 'becor':
                
                task_opt = self.optimizers()
                
                self.manual_backward(loss, task_opt)

                task_opt.step()

                task_opt.zero_grad()

            elif self.model_type == 'normal':

                task_opt, stain_opt = self.optimizers()

                self.manual_backward(torch.mean(task_loss), retain_graph = True)
                self.manual_backward(torch.mean(stain_loss))
                
                task_opt.step()
                stain_opt.step()

                task_opt.zero_grad()
                stain_opt.zero_grad()

        return loss
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        loss = self.cos_sim('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        loss = self.cos_sim('val', batch, batch_nb) 
        return loss


    def test_step(self, batch, batch_nb):
        # OPTIONAL
        loss = self.cos_sim('test', batch, batch_nb) 
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        return 

    def configure_optimizers(self):
        
        if self.model_type == 'normal':
            task_params = list(self.resnet.parameters()) + list(self.emb_2_task.parameters())  + list(self.classifier.parameters()) 
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

            stain_params = list(self.stain_classifier.parameters()) + list(self.emb_2_stain.parameters()) #list(self.resnet.parameters()) + 
            stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr) 

            return [task_opt, stain_opt]
            
        elif self.model_type == 'becor':
            task_params = self.parameters()        
            task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

            return [task_opt]
        
                        
        

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

class Bekind_indhe(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)

        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False

        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        self.classifier = nn.Linear(2048, self.hparams.num_classes)        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_task_f1"] = 0
            wandb.run.summary[who + "_wrst_task_f1"] = 1

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        
        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)

        stain_embeddings = self.non_lin(self.emb_2_stain(img_embs))
        task_embeddings = self.non_lin(self.emb_2_task(img_embs))

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        if self.model_type == 'becor':
            #Define some combination of task and source embeddings to be the final embeddings
            task_embeddings = task_embeddings - stain_embeddings
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Stain acc
        stain_preds = stain_logits.argmax(-1)        
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)
        wandb.run.summary[who + "_wrst_task_f1"]  = min(wandb.run.summary[who + "_wrst_task_f1"], task_f1)

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            self.manual_backward(torch.mean(task_loss), retain_graph = True)
            self.manual_backward(torch.mean(stain_loss))
            
            task_opt.step()
            stain_opt.step()

            task_opt.zero_grad()
            stain_opt.zero_grad()
            
        return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1}
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        results = self.cos_sim('train', batch, batch_nb)
        return results

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        results = self.cos_sim('val', batch, batch_nb) 
        return results

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        results = self.cos_sim('test', batch, batch_nb) 
        return results
    
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 

    def configure_optimizers(self):
        
        task_params = []
        for a in (self.resnet, self.emb_2_task, self.classifier):
            task_params += list(a.parameters())

        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

        stain_params = []
        for a in (self.resnet, self.stain_classifier, self.emb_2_stain):
            stain_params += list(a.parameters())

        stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr, weight_decay=self.weight_decay)
        
        return [task_opt, stain_opt]    
        

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

class Bekind_sl_best(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_sources', type=int, default=33)
        parser.add_argument('--num_slides', type=int, default=150)
        parser.add_argument('--srcs_map', type=dict)        
        parser.add_argument('--slides_map', type=dict)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        # self.resnet2 = nn.Sequential(*image_modules)
        
        self.emb_2_task = nn.Linear(2048, 2048)
        self.emb_2_src = nn.Linear(2048, 2048)
        self.emb_2_sl = nn.Linear(2048, 2048)
        
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.src_classifier = nn.Linear(2048, self.hparams.num_srcs)
        self.sl_classifier = nn.Linear(2048, self.hparams.num_slides)

        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.model_type = self.hparams.model_type

        self.non_lin = self.hparams.non_lin

    def forward(self, x):

        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))

        task_embs = self.emb_2_task(img_embs)
        src_embeddings = self.emb_2_src(img_embs)
        slide_embeddings = self.emb_2_sl(img_embs)

        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        #Introduce non-linearity
        task_embs, src_embeddings, slide_embeddings = (self.non_lin(x) for x in (task_embs, src_embeddings, slide_embeddings))

        return task_embs, src_embeddings, slide_embeddings

    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_label), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        task_embeddings, src_embeddings, slide_embeddings = self(x)
        
        #Define some combination of task and source embeddings to be the final embeddings
        task_embeddings = self.combine_embeddings(task_embeddings = task_embeddings, src_embeddings = src_embeddings,  slide_embeddings = slide_embeddings) 

        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        src_cos_sim = cos(task_embeddings, src_embeddings) 
        slide_cos_sim = cos(task_embeddings, slide_embeddings)
        slide_src_cos_sim = cos(src_embeddings, slide_embeddings)
        
        #Get the source from the slide id
        src_labels = torch.LongTensor([self.hparams.srcs_map[i[len('TCGA-') : len('TCGA-00')]] for i in slide_id]).to('cuda')
        slide_labels = torch.LongTensor([self.hparams.slides_map[i] for i in slide_id]).to('cuda')
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        src_logits = self.src_classifier(src_embeddings)
        slide_logits = self.sl_classifier(slide_embeddings)

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        src_loss = F.cross_entropy(src_logits, src_labels, reduction = "none") 
        slide_loss = F.cross_entropy(slide_logits, slide_labels, reduction = "none") 
                
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(   
                                    src_loss = src_loss,
                                    slide_loss = slide_loss,
                                    task_loss= task_loss,
                                    src_cos_sim= src_cos_sim,
                                    slide_cos_sim = slide_cos_sim,
                                    slide_src_cos_sim = slide_src_cos_sim
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)

        #Source acc
        src_preds = src_logits.argmax(-1)
        src_acc = torchmetrics.functional.accuracy(src_preds, src_labels)

        #Slide acc
        slide_preds = slide_logits.argmax(-1)
        slide_acc = torchmetrics.functional.accuracy(slide_preds, slide_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        src_f1 =  torchmetrics.functional.f1(src_preds, src_labels, num_classes = self.hparams.num_srcs, average = 'weighted')
        slide_f1 =  torchmetrics.functional.f1(slide_preds, slide_labels, num_classes = self.hparams.num_slides, average = 'weighted')

        self.log(who + '_task_f1', task_f1)
        self.log(who + '_src_f1', src_f1)
        self.log(who + '_slide_f1', slide_f1)

        self.log(who + '_src_cos_sim', torch.mean(src_cos_sim))
        self.log(who + '_abs_src_cos_sim', torch.mean(torch.abs(src_cos_sim)))        
        self.log(who + '_slide_cos_sim', torch.mean(slide_cos_sim))
        self.log(who + '_abs_slide_cos_sim', torch.mean(torch.abs(slide_cos_sim))) 
        self.log(who + '_slide_src_cos_sim', torch.mean(slide_src_cos_sim))
        self.log(who + '_abs_slide_src_cos_sim', torch.mean(torch.abs(slide_src_cos_sim)))        
        
        self.log(who + '_loss', loss)        
        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_src_loss', torch.mean(src_loss))
        self.log(who + '_slide_loss', torch.mean(slide_loss))

        self.log(who + '_task_acc', task_acc)
        self.log(who + '_src_acc', src_acc)
        self.log(who + '_slide_acc', slide_acc)

        if who == 'train':
            
            if self.model_type == 'becor':
                task_opt, _, _ = self.optimizers()

                self.manual_backward(loss, task_opt)

                task_opt.step()

            elif self.model_type == 'normal':
                task_opt, src_opt, slide_opt = self.optimizers()

                self.manual_backward(loss, retain_graph = True)
                self.manual_backward(torch.mean(src_loss), retain_graph=True)
                self.manual_backward(torch.mean(slide_loss))
                
                task_opt.step()
                src_opt.step()
                slide_opt.step()

        return loss
            

    def training_step(self, batch, batch_nb, optimizer_idx):
        # REQUIRED        
        loss = self.cos_sim('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        loss = self.cos_sim('val', batch, batch_nb) 
        return loss


    def test_step(self, batch, batch_nb):
        # OPTIONAL
        loss = self.cos_sim('test', batch, batch_nb) 
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        return 

    def configure_optimizers(self):
        
        if self.model_type == 'normal':
            task_params = list(self.resnet.parameters()) + list(self.classifier.parameters()) 
            src_params =  list(self.src_classifier.parameters()) + list(self.emb_2_src.parameters()) #+ list(self.resnet.parameters())            
            slide_params = list(self.sl_classifier.parameters()) + list(self.emb_2_sl.parameters()) #+ list(self.resnet.parameters())
            
        elif self.model_type == 'becor':
            task_params = self.parameters()
            src_params =  list(self.src_classifier.parameters()) + list(self.emb_2_src.parameters()) + list(self.resnet.parameters())            
            slide_params = list(self.sl_classifier.parameters()) + list(self.emb_2_sl.parameters()) + list(self.resnet.parameters())
            
        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)
        src_opt = torch.optim.Adam(src_params, lr=self.hparams.lr, weight_decay = self.hparams.weight_decay) 
        slide_opt = torch.optim.Adam(slide_params, lr=self.hparams.lr, weight_decay = self.hparams.weight_decay) 

        return [task_opt, src_opt, slide_opt]

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

class PretrainedResnet50FT_Best(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, _), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "mean")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', loss)
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class Bekind_indhe_cos_sim(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)
        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False

        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        self.classifier = nn.Linear(2048, self.hparams.num_classes)        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_task_f1"] = 0
            wandb.run.summary[who + "_wrst_task_f1"] = 1

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch
        
        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        task_stain_cos_sim = cos(task_embeddings, stain_embeddings) #torch.mean(torch.abs()) 
        
        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        # if self.model_type == 'becor':
        #     #Define some combination of task and source embeddings to be the final embeddings
        #     task_embeddings = task_embeddings - stain_embeddings
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        stain_embs_task_logits = self.classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        stain_embs_task_loss = F.cross_entropy(stain_embs_task_logits, task_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,                                    
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        #stain_embs_task
        stain_embs_task_preds = stain_embs_task_logits.argmax(-1)        
        
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)
        stain_embs_task_acc = torchmetrics.functional.accuracy(stain_embs_task_preds, task_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')
        stain_embs_task_f1 =  torchmetrics.functional.f1(stain_embs_task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)
        wandb.run.summary[who + "_wrst_task_f1"]  = min(wandb.run.summary[who + "_wrst_task_f1"], task_f1)

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        self.log(who + '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        self.log(who + '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        self.log(who + '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            self.manual_backward(torch.mean(task_loss), retain_graph = True)# - torch.mean(stain_embs_task_loss), retain_graph = True)
            self.manual_backward(torch.mean(stain_loss))
            
            task_opt.step()
            stain_opt.step()

            task_opt.zero_grad()
            stain_opt.zero_grad()
            
        return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        results = self.cos_sim('train', batch, batch_nb)
        return results

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        results = self.cos_sim('val', batch, batch_nb) 
        return results

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        results = self.cos_sim('test', batch, batch_nb) 
        return results
    
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 

    def configure_optimizers(self):
        
        task_params = []
        for a in (self.resnet, self.emb_2_task, self.classifier):
            task_params += list(a.parameters())

        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

        stain_params = []
        for a in (self.resnet, self.stain_classifier, self.emb_2_stain):
            stain_params += list(a.parameters())

        stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr, weight_decay=self.weight_decay)
        
        return [task_opt, stain_opt]    
        

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

class Bekind_indhe_dro(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)
        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False

        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        self.classifier = nn.Linear(2048, self.hparams.num_classes)        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_task_f1"] = 0
            wandb.run.summary[who + "_wrst_task_f1"] = 1

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

        
    def class_opt(self, who, i, j, batch):
        x, (task_labels, stain_labels), slide_id = batch
        
        task_labels_cpy = task_labels
        stain_labels_cpy = stain_labels

        task_labels = task_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        stain_labels = stain_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        
        num_class = len(task_labels)
        
        if num_class == 0:
            return 'num_class_0'
        
        self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

        img_embs = self(x)[(task_labels_cpy == i) & (stain_labels_cpy == j)]

        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)

        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)

        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 

        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')

        self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds.float()))
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_acc', stain_acc) #, on_step=True)

        self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + str(i) + '_' + str(j) + '_' '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            self.manual_backward(torch.mean(task_loss), retain_graph = True)
            self.manual_backward(torch.mean(stain_loss))
            
            task_opt.step()
            stain_opt.step()

            task_opt.zero_grad()
            stain_opt.zero_grad()
            
        # return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
        return task_f1
    
    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch
        
        f1 = torch.Tensor([1])

        for i in [0, 1]:
            for j in [0, 1]:
                task_f1 = self.class_opt(who, i, j, batch)
                
        return 

        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        task_stain_cos_sim = cos(task_embeddings, stain_embeddings) #torch.mean(torch.abs()) 
        
        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        # if self.model_type == 'becor':
        #     #Define some combination of task and source embeddings to be the final embeddings
        #     task_embeddings = task_embeddings - stain_embeddings
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        stain_embs_task_logits = self.classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        stain_embs_task_loss = F.cross_entropy(stain_embs_task_logits, task_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,                                    
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        #stain_embs_task
        stain_embs_task_preds = stain_embs_task_logits.argmax(-1)        
        
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)
        stain_embs_task_acc = torchmetrics.functional.accuracy(stain_embs_task_preds, task_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')
        stain_embs_task_f1 =  torchmetrics.functional.f1(stain_embs_task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)
        wandb.run.summary[who + "_wrst_task_f1"]  = min(wandb.run.summary[who + "_wrst_task_f1"], task_f1)

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        self.log(who + '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        self.log(who + '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        self.log(who + '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            self.manual_backward(torch.mean(task_loss) - torch.mean(stain_embs_task_loss), retain_graph = True)
            self.manual_backward(torch.mean(stain_loss))
            
            task_opt.step()
            stain_opt.step()

            task_opt.zero_grad()
            stain_opt.zero_grad()
            
        return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        results = self.cos_sim('train', batch, batch_nb)
        return results

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        results = self.cos_sim('val', batch, batch_nb) 
        return results

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        results = self.cos_sim('test', batch, batch_nb) 
        return results
    
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        
        # print(outputs)

        # self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 

    def configure_optimizers(self):
        
        task_params = []
        for a in (self.resnet, self.emb_2_task, self.classifier):
            task_params += list(a.parameters())

        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

        stain_params = []
        for a in (self.resnet, self.stain_classifier, self.emb_2_stain):
            stain_params += list(a.parameters())

        stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr, weight_decay=self.weight_decay)
        
        return [task_opt, stain_opt]    
        

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

class Bekind_indhe_dro_log(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)
        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False

        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        self.classifier = nn.Linear(2048, self.hparams.num_classes)        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_task_f1"] = 0
            wandb.run.summary[who + "_wrst_task_f1"] = 1

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

        
    def class_opt(self, who, i, j, batch):
        x, (task_labels, stain_labels), slide_id = batch
        
        task_labels_cpy = task_labels
        stain_labels_cpy = stain_labels

        task_labels = task_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        stain_labels = stain_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        
        num_class = len(task_labels)
        
        if num_class == 0:
            return 'num_class_0'
        
        self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

        img_embs = self(x)[(task_labels_cpy == i) & (stain_labels_cpy == j)]

        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)

        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)

        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 

        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')

        self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds.float()))
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_acc', stain_acc) #, on_step=True)

        self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + str(i) + '_' + str(j) + '_' '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        # if who == 'train':
            
        #     task_opt, stain_opt = self.optimizers()

        #     self.manual_backward(torch.mean(task_loss), retain_graph = True)
        #     self.manual_backward(torch.mean(stain_loss))
            
        #     task_opt.step()
        #     stain_opt.step()

        #     task_opt.zero_grad()
        #     stain_opt.zero_grad()
            
        # return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
        return 
    
    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch
                        
        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        task_stain_cos_sim = cos(task_embeddings, stain_embeddings) #torch.mean(torch.abs()) 
        
        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        if self.model_type == 'becor':
            #Define some combination of task and source embeddings to be the final embeddings
            task_embeddings = task_embeddings - stain_embeddings
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        stain_embs_task_logits = self.classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        stain_embs_task_loss = F.cross_entropy(stain_embs_task_logits, task_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,                                    
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        #stain_embs_task
        stain_embs_task_preds = stain_embs_task_logits.argmax(-1)        
        
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)
        stain_embs_task_acc = torchmetrics.functional.accuracy(stain_embs_task_preds, task_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')
        stain_embs_task_f1 =  torchmetrics.functional.f1(stain_embs_task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)
        wandb.run.summary[who + "_wrst_task_f1"]  = min(wandb.run.summary[who + "_wrst_task_f1"], task_f1)

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        self.log(who + '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        self.log(who + '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        self.log(who + '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]
                stain_logits_ij = stain_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]
                stain_loss_ij = stain_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)        
                
                #Stain acc
                stain_preds_ij = stain_logits_ij.argmax(-1)        

                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                stain_acc_ij = torchmetrics.functional.accuracy(stain_preds_ij, stain_labels_ij)

                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                stain_f1_ij =  torchmetrics.functional.f1(stain_preds_ij, stain_labels_ij, num_classes = self.hparams.num_stains, average = 'macro')

                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_pred', torch.mean(stain_preds_ij.float()))
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_stain_acc', stain_acc_ij) #, on_step=True)

                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_stain_f1', stain_f1_ij) #, on_step=True)

                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_stain_loss', torch.mean(stain_loss_ij)) #, on_step=True)

                task_accs.append(task_acc_ij)
                stain_accs.append(stain_acc_ij)
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))
        
        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            self.manual_backward(torch.mean(task_loss), retain_graph = True)
            self.manual_backward(torch.mean(stain_loss))
            
            task_opt.step()
            stain_opt.step()

            task_opt.zero_grad()
            stain_opt.zero_grad()
            
        return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        results = self.cos_sim('train', batch, batch_nb)
        return results

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        results = self.cos_sim('val', batch, batch_nb) 
        return results

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        results = self.cos_sim('test', batch, batch_nb) 
        return results
    
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        
        # print(outputs)

        # self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 

    def configure_optimizers(self):
        
        task_params = []
        for a in (self.resnet, self.emb_2_task, self.classifier):
            task_params += list(a.parameters())

        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

        stain_params = []
        for a in (self.resnet, self.stain_classifier, self.emb_2_stain):
            stain_params += list(a.parameters())

        stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr, weight_decay=self.weight_decay)
        
        return [task_opt, stain_opt]    
        

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

class PretrainedResnet50FT_Best_DRO_log(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        print('self.hparams.dropout', self.hparams.dropout)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        print(self.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
            
        #Define logits over the task and source embeddings
        embs = self(x)
        
        if who == 'train':
            embs = self.dropout(embs)       
        
        task_logits = self.classifier(embs)

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return torch.mean(loss)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_min(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        print('self.hparams.dropout', self.hparams.dropout)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        print(self.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []        
        task_losses = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                task_f1s.append(task_f1_ij)
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))

        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))

        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return max(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_mean_loss(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        min_loss = 1
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))
        
        recorded_loss = sum(task_losses) / len(task_losses)
        self.log(who + '_recorded_loss', recorded_loss)

        return sum(task_losses) / len(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class Bekind_indhe_dro_BOBW(pl.LightningModule):
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--num_stains', type=int, default=2)        
        parser.add_argument('--combine_loss')  
        parser.add_argument('--combine_embeddings')    
        parser.add_argument('--model_type')  
        parser.add_argument('--non_lin')
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--lr', type=float, default=1e-5)
        
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        
        #Freeze lower layers
        # for name, param in self.resnet.named_parameters():
        #     if name.split('.')[0] != '7':
        #         param.requires_grad = False

        self.model_type = self.hparams.model_type
        self.emb_2_stain = nn.Linear(2048, 2048)
        self.emb_2_task = nn.Linear(2048, 2048)

        self.classifier = nn.Linear(2048, self.hparams.num_classes)        
        self.stain_classifier = nn.Linear(2048, self.hparams.num_stains)
        
        self.combine_loss = self.hparams.combine_loss
        self.combine_embeddings = self.hparams.combine_embeddings
        
        self.non_lin = self.hparams.non_lin
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.weight_decay = self.hparams.weight_decay
        
        for who in ['train', 'val', 'test']:
            wandb.run.summary[who + "_best_task_f1"] = 0
            wandb.run.summary[who + "_wrst_task_f1"] = 1

    def forward(self, x):
        
        
        #Get the embeddings for the task and source
        img_embs = self.non_lin(torch.flatten(self.resnet(x), 1))
        
        # self.log(who + '_src_embs_norm', torch.mean(LA.norm(src_embeddings, dim=1))) 
        # self.log(who + '_img_embs_norm', torch.mean(LA.norm(img_embs, dim=1))) 

        return img_embs

        
    def class_opt(self, who, i, j, batch):
        x, (task_labels, stain_labels), slide_id = batch
        
        task_labels_cpy = task_labels
        stain_labels_cpy = stain_labels

        task_labels = task_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        stain_labels = stain_labels_cpy[(task_labels_cpy == i) & (stain_labels_cpy == j)]
        
        num_class = len(task_labels)
        
        if num_class == 0:
            return 'num_class_0'
        
        self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

        img_embs = self(x)[(task_labels_cpy == i) & (stain_labels_cpy == j)]

        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)

        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)

        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 

        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')

        self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds.float()))
        self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_acc', stain_acc) #, on_step=True)

        self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
        self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + str(i) + '_' + str(j) + '_' '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + str(i) + '_' + str(j) + '_' '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        # self.log(who + str(i) + '_' + str(j) + '_' '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        # if who == 'train':
            
        #     task_opt, stain_opt = self.optimizers()

        #     self.manual_backward(torch.mean(task_loss), retain_graph = True)
        #     self.manual_backward(torch.mean(stain_loss))
            
        #     task_opt.step()
        #     stain_opt.step()

        #     task_opt.zero_grad()
        #     stain_opt.zero_grad()
            
        # return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
        return 
    
    def cos_sim(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch
                        
        img_embs = self(x)
        
        if who == 'train':
            img_embs = self.dropout(img_embs)
        
        stain_embeddings = self.emb_2_stain(img_embs)
        task_embeddings = self.emb_2_task(img_embs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)    
        task_stain_cos_sim = cos(task_embeddings, stain_embeddings) #torch.mean(torch.abs()) 
        
        stain_embeddings = self.non_lin(stain_embeddings)
        task_embeddings = self.non_lin(task_embeddings)

        if who == 'train':
            stain_embeddings = self.dropout(stain_embeddings)
            task_embeddings = self.dropout(task_embeddings)
        
        # if self.model_type == 'becor':
        #     #Define some combination of task and source embeddings to be the final embeddings
        #     task_embeddings = task_embeddings - stain_embeddings
        
        #Define cosine similarity over the two embeddings
        #Rationalie is that the task embeddings are really similar to the source embeddings and we want to penalize this
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(task_embeddings)
        stain_logits = self.stain_classifier(stain_embeddings)
        stain_embs_task_logits = self.classifier(stain_embeddings)
        
        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        stain_loss = F.cross_entropy(stain_logits, stain_labels, reduction = "none") 
        stain_embs_task_loss = F.cross_entropy(stain_embs_task_logits, task_labels, reduction = "none") 
        
        #Combine the loss values somehow using a predefined function
        loss = self.combine_loss(  
                                    stain_loss = stain_loss,
                                    task_loss= task_loss,                                    
                                )
        
        #Train acc
        task_preds = task_logits.argmax(-1)        
        
        #Stain acc
        stain_preds = stain_logits.argmax(-1)        

        #stain_embs_task
        stain_embs_task_preds = stain_embs_task_logits.argmax(-1)        
        
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        stain_acc = torchmetrics.functional.accuracy(stain_preds, stain_labels)
        stain_embs_task_acc = torchmetrics.functional.accuracy(stain_embs_task_preds, task_labels)

        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        stain_f1 =  torchmetrics.functional.f1(stain_preds, stain_labels, num_classes = self.hparams.num_stains, average = 'macro')
        stain_embs_task_f1 =  torchmetrics.functional.f1(stain_embs_task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')

        wandb.run.summary[who + "_best_task_f1"]  = max(wandb.run.summary[who + "_best_task_f1"], task_f1)
        wandb.run.summary[who + "_wrst_task_f1"]  = min(wandb.run.summary[who + "_wrst_task_f1"], task_f1)

        self.log(who + '_av_label', torch.mean(task_labels.float())) #, on_step=True)
        self.log(who + '_av_stain_label', torch.mean(stain_labels.float())) #, on_step=True)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        self.log(who + '_av_stain_pred', torch.mean(stain_preds.float()))
        
        self.log(who + '_task_f1', task_f1) #, on_step=True)
        self.log(who + '_stain_f1', stain_f1) #, on_step=True)

        # self.log(who + '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
        # self.log(who + '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
        
        self.log(who + '_loss', loss) #, on_step=True)        
        self.log(who + '_task_loss', torch.mean(task_loss)) #, on_step=True)
        self.log(who + '_stain_loss', torch.mean(stain_loss)) #, on_step=True)
        # self.log(who + '_task_stain_cos_sim', torch.mean(task_stain_cos_sim)) #, on_step=True)
        
        self.log(who + '_stain_embs_task_loss', torch.mean(stain_embs_task_loss)) 
        self.log(who + '_stain_embs_task_acc', torch.mean(stain_embs_task_acc)) 
        self.log(who + '_stain_embs_task_f1', torch.mean(stain_embs_task_f1)) 

        self.log(who + '_task_acc', task_acc) #, on_step=True)
        self.log(who + '_stain_acc', stain_acc) #, on_step=True)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))

        if who == 'train':
            
            task_opt, stain_opt = self.optimizers()

            # self.manual_backward(torch.mean(task_loss), retain_graph = True)
            
            # task_opt.step()

            # task_opt.zero_grad()

            #torch.tensor(torch.mean(torch.tensor(task_losses)), requires_grad=True)

            self.manual_backward(torch.tensor(torch.max(torch.tensor(task_losses)), requires_grad=True))
            
            task_opt.step()

            task_opt.zero_grad()

            
        return {'loss' : loss, 'task_acc' : task_acc, 'task_f1' : task_f1, 'task_stain_cos_sim' : task_stain_cos_sim}
            
    def training_step(self, batch, batch_nb, optimizer_idx = None):
        # REQUIRED        
        results = self.cos_sim('train', batch, batch_nb)
        return results

    def validation_step(self, batch, batch_nb):
        # OPTIONAL        
        results = self.cos_sim('val', batch, batch_nb) 
        return results

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        results = self.cos_sim('test', batch, batch_nb) 
        return results
    
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        
        # print(outputs)

        # self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 

    def configure_optimizers(self):
        
        task_params = []
        for a in (self.resnet, self.emb_2_task, self.classifier):
            task_params += list(a.parameters())

        task_opt = torch.optim.Adam(task_params, lr=self.hparams.lr, weight_decay=self.weight_decay)

        stain_params = []
        for a in (self.resnet, self.stain_classifier, self.emb_2_stain):
            stain_params += list(a.parameters())

        stain_opt = torch.optim.Adam(stain_params, lr=self.hparams.lr, weight_decay=self.weight_decay)
        
        return [task_opt, stain_opt]    
        

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

class PretrainedResnet50FT_Best_DRO_1_over_n(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--C', type=float, default=1e-3)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.C = self.hparams.C

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        min_loss = 1
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij) + self.C / math.sqrt(num_class))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))
        
        recorded_loss = torch.tensor(torch.max(torch.tensor(task_losses)), requires_grad=True)
        self.log(who + '_recorded_loss', recorded_loss)

        print(task_loss)
        print(recorded_loss)
        
        return recorded_loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_mean_and_worst(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue

                class_sizes.append(num_class)
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        
        loss = torch.mean(task_loss) + (min(class_sizes)) * torch.tensor(torch.max(torch.tensor(task_losses)), requires_grad=True)

        self.log(who + '_loss', loss)
        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet18FT(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--C', type=float, default=1e-3)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet18(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(512, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.C = self.hparams.C

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        min_loss = 1
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij) + self.C / math.sqrt(num_class))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))

        return torch.tensor(torch.max(torch.tensor(task_losses)), requires_grad=True)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet18FT_random_matrix(pl.LightningModule):

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--C', type=float, default=1e-3)
        parser.add_argument('--M', type=int, default=64)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet18(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(self.hparams.M, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.C = self.hparams.C
        self.random_matrix = torch.rand(512, self.hparams.M).cuda()

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Random Projection 
        rp = torch.matmul(self(x), self.random_matrix)

        #Define logits over the task and source embeddings
        task_logits = self.classifier(rp)

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        min_loss = 1
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij) + self.C / math.sqrt(num_class))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return torch.mean(task_loss)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_random_matrix(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--C', type=float, default=1e-3)
        parser.add_argument('--M', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.2)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(self.hparams.M, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.C = self.hparams.C
        self.random_matrix = torch.rand(2048, self.hparams.M).cuda()

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Random Projection 
        rp = torch.matmul(self(x), self.random_matrix)

        #Define logits over the task and source embeddings
        task_logits = self.classifier(rp)

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        min_loss = 1
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                class_sizes.append(num_class)

                if num_class == 0:
                    continue
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij) + self.C / math.sqrt(num_class))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return torch.mean(task_loss)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_worst_of_batch(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')

        self.log(who + '_task_loss', torch.mean(task_loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        class_sizes = []
        
        # print(torch.mean(loss))

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue

                class_sizes.append(num_class)
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'macro')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))
        
        min_loss = torch.tensor(torch.min(torch.tensor(task_losses)), requires_grad=True)
        max_loss = torch.tensor(torch.max(torch.tensor(task_losses)), requires_grad=True)
        
        loss = torch.max(task_loss)

        self.log(who + '_loss', loss)
        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Min_Loss_Debug(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        print('self.hparams.dropout', self.hparams.dropout)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        print(self.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        stain_accs = []
        task_losses = []
        num_in_each_class = []

        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                else:
                    num_in_each_class.append(num_class)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))


        return task_losses[min(enumerate(num_in_each_class), key=lambda x: x[1])]

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_log(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.max_val_f1 = 0

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return out

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])
        uniq_srcs = set([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])
        
        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in uniq_srcs:                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_num_hosps', len(set(srcs)))
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def validation_epoch_end(self, outputs):
        
        val_f1 = torch.mean(torch.tensor([output['task_f1'] for output in outputs]))
        self.max_val_f1 = max(self.max_val_f1, val_f1)        
        self.log('best_val_f1', self.max_val_f1)
        # wandb.run.summary["val_best_task_f1"]  = max(wandb.run.summary["val_best_task_f1"], val_f1)
        
        return 
    
    def test_epoch_end(self, outputs):
        
        # self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_mean(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return sum(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_max(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_num_hosps', len(set(srcs)))
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return max(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_gap(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return max(task_losses) - min(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_weighted(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []
        avg_len = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            avg_len.append(num_class)

            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append( torch.mean(task_loss_src) * 1 / num_class )
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return sum(task_losses) / len(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_plus_1_over_n(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        out = self.dropout(out)       
        return out

    def step(self, who, batch, batch_nb):    
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        # task_accs = torch.Tensor([])
        # task_f1s = torch.Tensor([])
        # task_losses = torch.Tensor([])  

        task_accs = []
        task_f1s = []
        task_losses = []
        avg_len = []

        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            avg_len.append(num_class)

            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            # print(task_accs)
            # print(task_acc_src)

            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, torch.Tensor([task_f1_src])))
            # task_losses = torch.cat((task_losses, torch.Tensor([task_loss_src])))
            
            # task_accs = torch.cat((task_accs, task_acc_src))
            # task_f1s = torch.cat((task_f1s, task_f1_src))
            # task_losses = torch.cat((task_losses, task_loss_src))

            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src) + self.hparams.C / np.sqrt(num_class))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return sum(task_losses) / len(task_losses)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_abstain_old(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.max_val_f1 = 0
        
    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return out

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)
        
        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
                
        confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
        num_confident = len(task_logits[confidence_region])

        self.log(who + '_num_confident', num_confident)

        if num_confident > 0:
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]

        """        
        if self.global_step > 50:
            confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]
        """
        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
    
    def validation_epoch_end(self, outputs):
        
        val_f1 = torch.mean(torch.tensor([output['task_f1'] for output in outputs]))
        self.max_val_f1 = max(self.max_val_f1, val_f1)        
        self.log('best_val_f1', self.max_val_f1)
        
        return 

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_abstain(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label, _ in valid_loader:
                input = input.cuda()
                logits = self(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        self.log('before_temperature_nll', before_temperature_nll)
        self.log('before_temperature_ece', before_temperature_ece)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        self.log('Optimal temperature', self.temperature.item())
        self.log('after_temperature_nll', after_temperature_nll)
        self.log('after_temperature_ece', after_temperature_ece)

        return self

    def step(self, who, batch, batch_nb):    
            
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
                
        confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
        
        num_confident = len(task_logits[confidence_region])

        self.log(who + '_num_confident', num_confident)

        if num_confident > 0:
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            stain_labels = stain_labels[confidence_region]
            
        """        
        if self.global_step > 50:
            confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]
        """

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                task_f1s.append(task_f1_ij)
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
    
    def validation_epoch_end(self, outputs):
        
        val_f1 = torch.mean(torch.tensor([output['task_f1'] for output in outputs]))
        self.max_val_f1 = max(self.max_val_f1, val_f1)        
        self.log('best_val_f1', self.max_val_f1)
        
        return 

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Best_DRO_abstain_conservative(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return out

    def step(self, who, batch, batch_nb):    
            
        x, (task_labels, stain_labels), slide_id = batch

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(self(x))

        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
                
        confidence_region = (probabilities[:, 0] < self.hparams.confidence_threshold) & (probabilities[:, 1] < self.hparams.confidence_threshold)
        
        num_confident = len(task_logits[confidence_region])

        self.log(who + '_num_confident', num_confident)

        if num_confident > 0:
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            stain_labels = stain_labels[confidence_region]

        """        
        if self.global_step > 50:
            confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]
        """

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for i in [0, 1]:
            for j in [0, 1]:        
                
                task_labels_ij = task_labels[(task_labels == i) & (stain_labels == j)]
                stain_labels_ij = stain_labels[(task_labels == i) & (stain_labels == j)]
                
                num_class = len(task_labels_ij)
                
                if num_class == 0:
                    continue
                
                self.log(who + str(i) + '_' + str(j) + '_' '_len', num_class)

                task_logits_ij = task_logits[(task_labels == i) & (stain_labels == j)]

                task_loss_ij = task_loss[(task_labels == i) & (stain_labels == j)]

                #Train acc
                task_preds_ij = task_logits_ij.argmax(-1)                        
                
                task_acc_ij = torchmetrics.functional.accuracy(task_preds_ij, task_labels_ij)
                
                #F1
                task_f1_ij = torchmetrics.functional.f1(task_preds_ij, task_labels_ij, num_classes = self.hparams.num_classes, average = 'weighted')
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_av_label', torch.mean(task_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_stain_label', torch.mean(stain_labels_ij.float())) #, on_step=True)
                self.log(who + str(i) + '_' + str(j) + '_' '_av_pred', torch.mean(task_preds_ij.float()))
                                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_acc', task_acc_ij) #, on_step=True)
                
                self.log(who + str(i) + '_' + str(j) + '_' '_task_f1', task_f1_ij) #, on_step=True)
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
                # self.log(who + str(i) + '_' + str(j) + '_' '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
                
                # self.log(who + str(i) + '_' + str(j) + '_' '_loss', loss) #, on_step=True)        
                self.log(who + str(i) + '_' + str(j) + '_' '_task_loss', torch.mean(task_loss_ij)) #, on_step=True)
                
                task_accs.append(task_acc_ij)
                task_f1s.append(task_f1_ij)
                task_losses.append(torch.mean(task_loss_ij))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_abstain_conservative(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return out

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)
        
        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
                
        confidence_region = (probabilities[:, 0] < self.hparams.confidence_threshold) & (probabilities[:, 1] < self.hparams.confidence_threshold)
        
        num_confident = len(task_logits[confidence_region])

        self.log(who + '_num_confident', num_confident)

        if num_confident > 0:
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]

        """        
        if self.global_step > 50:
            confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
            task_logits = task_logits[confidence_region]
            task_labels = task_labels[confidence_region]
            srcs = srcs[confidence_region.cpu()]
        """
        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_log_two_test(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return out

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])
        uniq_srcs = set([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        for src in uniq_srcs:
            if src not in self.hparams.srcs_map:
                who = 'external_test'
        
        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)

        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'weighted')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in uniq_srcs:                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'weighted')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_classes_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        self.log(who + '_num_hosps', len(set(srcs)))
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

        
    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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

class PretrainedResnet50FT_Hosp_DRO_abstain(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
        self.resnet = nn.Sequential(*image_modules)
        self.classifier = nn.Linear(2048, self.hparams.num_classes)
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.max_val_f1 = 0
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        out = self.resnet(x)
        out = torch.flatten(out, 1) 
        return self.temperature_scale(out)
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label, _ in valid_loader:
                input = input.cuda()
                logits = self(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    def step(self, who, batch, batch_nb):    
            
        x, task_labels, slide_id = batch

        srcs = np.array([i[len('TCGA-') : len('TCGA-00')] for i in slide_id])

        self.log(who + '_av_label', torch.mean(task_labels.float()))
        
        embs = self(x)

        if who == 'train':
            embs = self.dropout(embs)       
        
        #Define logits over the task and source embeddings
        task_logits = self.classifier(embs)
        
        #Converting logits to probabilities
        sm = torch.nn.Softmax()
        probabilities = sm(task_logits) 
        
        # confidence_region = torch.any(probabilities > self.hparams.confidence_threshold, 1)
        # confidence_region = (probabilities[:, 1] > self.hparams.confidence_threshold) | (probabilities[:, 0] > self.hparams.confidence_threshold)
        
        confidence_region = torch.any(probabilities > self.hparams.confidence_threshold, 1)

        num_confident = len(task_logits[confidence_region])

        if self.hparams.include_all_val == 'False':            

            self.log(who + '_num_confident', num_confident)

            if num_confident > 0:
                task_logits = task_logits[confidence_region]
                task_labels = task_labels[confidence_region]
                srcs = srcs[confidence_region.cpu()]
        
        elif self.hparams.include_all_val == 'True':
            if who == 'train': 

                num_confident = len(task_logits[confidence_region])

                self.log(who + '_num_confident', num_confident)

                if num_confident > 0:
                    task_logits = task_logits[confidence_region]
                    task_labels = task_labels[confidence_region]
                    srcs = srcs[confidence_region.cpu()]
        
        #Define loss values over the logits
        loss = task_loss = F.cross_entropy(task_logits, task_labels, reduction = "none")                
        
        if (who == 'train') and (self.hparams.include_num_confident == 'True') :
            loss += 1/(num_confident + 1)
                
        #Train acc
        task_preds = task_logits.argmax(-1)
        self.log(who + '_av_pred', torch.mean(task_preds.float()))
        task_acc = torchmetrics.functional.accuracy(task_preds, task_labels)
        
        #F1
        task_f1 = torchmetrics.functional.f1(task_preds, task_labels, num_classes = self.hparams.num_classes, average = 'macro')
        
        self.log(who + '_task_loss', torch.mean(loss))
        self.log(who + '_task_acc', task_acc)
        self.log(who + '_task_f1', task_f1)

        #DRO Logging
        task_accs = []
        task_f1s = []
        task_losses = []
        for src in set(srcs):                        
            task_labels_src = task_labels[srcs == src]
            
            num_class = len(task_labels_src)
            
            if num_class == 0:
                continue
            
            # self.log(who + src +  '_len', num_class)

            task_logits_src = task_logits[srcs == src]

            task_loss_src = task_loss[srcs == src]

            #Train acc
            task_preds_src = task_logits_src.argmax(-1)                        
            
            task_acc_src = torchmetrics.functional.accuracy(task_preds_src, task_labels_src)
            
            #F1
            task_f1_src = torchmetrics.functional.f1(task_preds_src, task_labels_src, num_classes = self.hparams.num_classes, average = 'macro')
            
            # self.log(who + src +  '_av_label', torch.mean(task_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_stain_label', torch.mean(stain_labels_src.float())) #, on_step=True)
            # self.log(who + src +  '_av_pred', torch.mean(task_preds_src.float()))
                            
            # self.log(who + src +  '_task_acc', task_acc_src) #, on_step=True)
            
            # self.log(who + src +  '_task_f1', task_f1_src) #, on_step=True)
            
            # self.log(who + src +  '_stain_task_cos_sim', torch.mean(stain_task_cos_sim))
            # self.log(who + src +  '_abs_stain_task_cos_sim', torch.mean(torch.abs(stain_task_cos_sim)))        
            
            # self.log(who + src +  '_loss', loss) #, on_step=True)        
            # self.log(who + src +  '_task_loss', torch.mean(task_loss_src)) #, on_step=True)
            
            task_accs.append(task_acc_src)
            task_f1s.append(task_f1_src)            
            task_losses.append(torch.mean(task_loss_src))
            
        self.log(who + '_min_task_acc', torch.min(torch.tensor(task_accs)))
        self.log(who + '_max_task_acc', torch.max(torch.tensor(task_accs)))
        self.log(who + '_max-min_task_acc', torch.max(torch.tensor(task_accs)) - torch.min(torch.tensor(task_accs)))
        self.log(who + '_min_task_f1', torch.min(torch.tensor(task_f1s)))
        self.log(who + '_max_task_f1', torch.max(torch.tensor(task_f1s)))
        self.log(who + '_max-min_task_f1', torch.max(torch.tensor(task_f1s)) - torch.min(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_f1', torch.mean(torch.tensor(task_f1s)))
        self.log(who + '_mean_task_acc', torch.mean(torch.tensor(task_accs)))
        self.log(who + '_max_task_loss', torch.max(torch.tensor(task_losses)))
        self.log(who + '_min_task_loss', torch.min(torch.tensor(task_losses)))
        self.log(who + '_mean_srcs_task_loss', torch.mean(torch.tensor(task_losses)))
        self.log(who + '_max-min_task_loss', torch.max(torch.tensor(task_losses)) - torch.min(torch.tensor(task_losses)))
        spearman = SpearmanCorrcoef()
        # self.log(who + '_task_stain_label_corr', spearman(task_labels.float(), stain_labels.float()))

        return {'loss' : torch.mean(loss), 'task_acc' : task_acc, 'task_f1' : task_f1, 'av_label': torch.mean(task_labels.float())} 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss
    
    def validation_epoch_end(self, outputs):
        
        val_f1 = torch.mean(torch.tensor([output['task_f1'] for output in outputs]))
        self.max_val_f1 = max(self.max_val_f1, val_f1)        
        self.log('best_val_f1', self.max_val_f1)
        
        return 

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def test_epoch_end(self, outputs):
        
        print(outputs)

        self.log( 'test_batch_acc_std', torch.std(torch.tensor([output['task_acc'] for output in outputs])) )
        
        return 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

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
