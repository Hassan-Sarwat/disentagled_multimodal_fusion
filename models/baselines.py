import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, MeanMetric
from losses import AvgTrustedLoss, SingleEvidentialLoss
from utils import get_cml_fusion, get_avg_fusion, discounted_belief_fusion
from classifiers import EvidentialNN
from torch import nn
from common_fusions import Concat


class LateFusion(pl.LightningModule):
    def __init__(self, feature_encoders, output_dims = [100,100],num_classes=42, dropout=0.3, aggregation='cml', lr=1e-4, annealing_start=20,
                 hidden_dim=(32), optimizer=torch.optim.Adam, weight_decay=1e-5, fused=1):
        super(LateFusion, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.fused  = fused
        self.feature_encoders = nn.ModuleList([i[0](**i[1]) for i in feature_encoders])
        self.weight_decay = weight_decay
        
        self.heads = nn.ModuleList([
            EvidentialNN(dropout=dropout, output_dims=num_classes, layers=(output_dims[i], *hidden_dim))
            for i in range(len(feature_encoders))])
        self.aggregation = {'cml':get_cml_fusion, 
                            'avg':get_avg_fusion,
                            'dbf':discounted_belief_fusion}[aggregation]
            
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = AvgTrustedLoss(num_views=len(feature_encoders), annealing_start=annealing_start)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes) for _ in feature_encoders])
        self.val_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes) for _ in feature_encoders])
        self.test_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes) for _ in feature_encoders])

    def forward(self, inputs):
        evidences = []
        for i, feature_encoder in enumerate(self.feature_encoders):
            output = feature_encoder(inputs[i].float())
            evidences.append(self.heads[i](output))
        return evidences

    def training_step(self, batch, batch_idx):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.log('train_loss', loss)
        acc = self.train_acc(evidences_a, target)
        self.log('train_acc_step', acc, prog_bar=True)

        # Per-modality accuracies
        for i, modality_acc in enumerate(self.train_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)
            self.log(f'train_acc_modality_{i}', modality_acc.compute(), prog_bar=False)
        return loss

    def shared_step(self, batch):
        target = batch[-1]
        evidences = self(batch)
        evidences = torch.stack(
             evidences, dim=1
        )
        evidences_a = self.aggregation(evidences)
        loss = self.criterion(evidences, target, evidences_a, fused=self.fused)
        return loss, evidences_a, target, evidences

    def validation_step(self, batch, batch_idx):
        loss, output, target, evidences = self.shared_step(batch)
        self.val_acc(output, target)
        # Per-modality validation accuracies
        for i, modality_acc in enumerate(self.val_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        alphas = output + 1
        denominator = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denominator
        entropy = self.num_classes / denominator
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denominator + 1)), dim=-1)
        self.validation_step_outputs.append({'loss': loss.detach(),'entropy': entropy.detach(),'aleatoric': aleatoric_uncertainty.detach()})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, output, target, evidences = self.shared_step(batch)
        self.test_acc(output, target)
        # Per-modality validation accuracies
        for i, modality_acc in enumerate(self.test_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        alphas = output + 1
        denominator = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denominator
        entropy = self.num_classes / denominator
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denominator + 1)), dim=-1)
        self.test_step_outputs.append({'loss': loss.detach(),'entropy': entropy.detach(),'aleatoric': aleatoric_uncertainty.detach()})
 
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True) 
        self.criterion.annealing_step += 1
        

    def on_validation_epoch_end(self):
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        entropies = torch.cat([x['entropy'] for x in self.validation_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.validation_step_outputs])
        
        self.log('val_acc', self.val_acc.compute(), on_step=False, prog_bar=True)
        self.log('val_loss', losses.mean(), on_step=False, prog_bar=True)
        self.log('val_entropy', entropies.mean(), on_step=False, prog_bar=True)
        self.log('val_sigma', aleatorics.mean(), on_step=False, prog_bar=True)

        # Per-modality accuracy logging
        for i, modality_acc in enumerate(self.val_modality_accs):
            self.log(f'val_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()


        self.val_acc.reset()
        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        entropies = torch.cat([x['entropy'] for x in self.test_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.test_step_outputs])

        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', entropies.mean())
        self.log('test_ale', aleatorics.mean())

        # Per-modality accuracy logging
        for i, modality_acc in enumerate(self.test_modality_accs):
            self.log(f'test_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

        self.test_acc.reset()  # Optional, 
        self.test_step_outputs.clear() 

    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class IntermediateFusion(pl.LightningModule):
    def __init__(self, feature_encoders, fusion='concat', output_dims = [100,100],num_classes=42, dropout=0.3, lr=1e-4, annealing_start=20,
                 hidden_dim=32, optimizer=torch.optim.Adam):
        super(IntermediateFusion, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.feature_encoders = nn.ModuleList([i[0](**i[1]) for i in feature_encoders])
        self.fusion = {'concat':Concat()}[fusion]
        if fusion=='concat':
            output_dim = sum(output_dims)
        self.head = EvidentialNN(dropout=dropout, output_dims=num_classes, layers=(output_dim, hidden_dim))
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = SingleEvidentialLoss(annealing_start=annealing_start)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs):
        outputs = []
        for i, feature_encoder in enumerate(self.feature_encoders):
            outputs.append(feature_encoder(inputs[i].float()))
        
        fused = self.fusion(outputs)
        evidences = self.head(fused)
        return evidences

    def training_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        target = batch[-1]
        evidence = self(batch)
        loss = self.criterion(evidence, target)
        return loss, evidence, target

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.val_acc(output, target)
        alphas = output + 1
        denominator = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denominator
        entropy = self.num_classes / denominator
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denominator + 1)), dim=-1)
        self.validation_step_outputs.append({'loss': loss.detach(),'entropy': entropy.detach(),'aleatoric': aleatoric_uncertainty.detach()})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, output, target = self.shared_step(batch)
        self.test_acc(output, target)
        alphas = output + 1
        denominator = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denominator
        entropy = self.num_classes / denominator
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denominator + 1)), dim=-1)
        self.test_step_outputs.append({'loss': loss.detach(),'entropy': entropy.detach(),'aleatoric': aleatoric_uncertainty.detach()})
 
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True) 
        self.criterion.annealing_step += 1

    def on_validation_epoch_end(self):
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        entropies = torch.cat([x['entropy'] for x in self.validation_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.validation_step_outputs])
        
        self.log('val_acc', self.val_acc.compute(), on_step=False, prog_bar=True)
        self.log('val_loss', losses.mean(), on_step=False, prog_bar=True)
        self.log('val_entropy', entropies.mean(), on_step=False, prog_bar=True)
        self.log('val_sigma', aleatorics.mean(), on_step=False, prog_bar=True)

        self.val_acc.reset()
        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        entropies = torch.cat([x['entropy'] for x in self.test_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.test_step_outputs])

        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', entropies.mean())
        self.log('test_ale', aleatorics.mean())

        self.test_acc.reset()  # Optional, 
        self.test_step_outputs.clear() 

    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    

