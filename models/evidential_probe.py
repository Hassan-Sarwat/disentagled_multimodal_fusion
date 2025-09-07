import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from losses import AvgTrustedLoss
from utils import get_cml_fusion, get_avg_fusion, get_joint_fusion, get_disentangled_fusion
from classifiers import EvidentialNN
import copy


class EvidentialProbeModule(pl.LightningModule):
    def __init__(self,backbone,num_classes,input_dim,hidden_dim=(32),lr=1e-4,dropout=0.3,annealing_start=20,
                 optimizer=torch.optim.Adam,freeze_backbone=True,aggregation='cml', fused=1):
        super().__init__()
        # Backbone & modality count
        self.backbone = copy.deepcopy(backbone)
        if not hasattr(self.backbone, 'N'):
            raise ValueError("backbone must expose attribute 'N' (number of modalities).")
        self.N = int(self.backbone.N)
        self.fused = fused
        # Views = 1 (shared) + N (specific per modality)
        self.num_views = 1 + self.N
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.annealing_start = annealing_start

        # Fusion/aggregation (assumed N-agnostic)
        self.agg = {
            'cml': get_cml_fusion,
            'avg': get_avg_fusion,
            'joint': get_joint_fusion,
            'disentangled': get_disentangled_fusion
        }[aggregation]

        # Probes: 1 shared + N specific
        self.x_shared = EvidentialNN(dropout=dropout, output_dims=num_classes,
                                     layers=(input_dim, *hidden_dim))
        self.x_specs = nn.ModuleList([
            EvidentialNN(dropout=dropout, output_dims=num_classes, layers=(input_dim, *hidden_dim))
            for _ in range(self.N)
        ])

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        # Per-view (shared + specifics) accuracies
        self.train_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                  for _ in range(self.num_views)])
        self.val_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                for _ in range(self.num_views)])
        self.test_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                 for _ in range(self.num_views)])

        # Loss (assumed supports arbitrary num_views)
        self.criterion = AvgTrustedLoss(num_views=self.num_views, annealing_start=annealing_start)

        if freeze_backbone:
            self.freeze_backbone()

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def get_embedding(self, batch):
        """
        Returns normalized embeddings:
          Zc: (B, N*input_dim)
          Zp_list: list of length N with tensors (B, input_dim)
        Accepts batch as [x0, ..., x_{N-1}, y] or [x0, ..., x_{N-1}]
        """
        xs = batch
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            last = batch[-1]
            if torch.is_tensor(last) and last.dtype in (torch.int64, torch.int32) and last.ndim >= 1:
                xs = batch[:-1]
        Zc_concat, Zp_list = self.backbone.get_embedding(xs)  # expects (concat_shared, list_of_specific)
        return Zc_concat, Zp_list

    def forward(self, x):
        """
        x: batch list; returns list of evidences [shared, spec_0, ..., spec_{N-1}]
        Each evidence is (B, num_classes).
        """
        Zc, Zp_list = self.get_embedding(x)
        evid_shared = self.x_shared(Zc)
        evid_specs = [self.x_specs[i](Zp_list[i]) for i in range(self.N)]
        return [evid_shared] + evid_specs

    def shared_step(self, batch):
        labels = batch[-1]
        evidences_list = self(batch)                    # list length = 1 + N
        evidences = torch.stack(evidences_list, dim=1)  # (B, num_views, num_classes)
        evidences_a = self.agg(evidences)               # (B, num_classes), fusion is N-agnostic
        loss = self.criterion(evidences, labels, evidences_a, fused=self.fused)
        return loss, evidences_a, labels, evidences

    # ----------------- Training -----------------
    def training_step(self, batch, batch_idx=None):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Overall accuracy (torchmetrics accepts logits/probs; evidential "evidence" works via argmax)
        acc = self.train_acc(evidences_a, target)
        self.log('train_acc_step', acc, on_step=False, on_epoch=True, prog_bar=True)

        # Per-view accuracies (shared is index 0)
        for i, modality_acc in enumerate(self.train_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)
            self.log(f'train_acc_modality_{i}_step', modality_acc.compute(), prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.criterion.annealing_step += 1
        for i, modality_acc in enumerate(self.train_modality_accs):
            self.log(f'train_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

    # ----------------- Validation -----------------
    def validation_step(self, batch, batch_idx):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.val_acc(evidences_a, target)

        for i, modality_acc in enumerate(self.val_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        # Uncertainty summaries
        alphas = evidences_a + 1
        denom = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denom
        entropy = self.num_classes / denom
        aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denom + 1)), dim=-1)

        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'entropy': entropy.detach(),
            'aleatoric': aleatoric.detach()
        })

    def on_validation_epoch_end(self):
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        entropies = torch.cat([x['entropy'] for x in self.validation_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.validation_step_outputs])

        self.log('val_loss', losses.mean(), on_step=False, prog_bar=True)
        self.log('val_entropy', entropies.mean(), on_step=False, prog_bar=True)
        self.log('val_sigma', aleatorics.mean(), on_step=False, prog_bar=True)
        self.log('val_acc', self.val_acc.compute(), on_step=False, prog_bar=True)

        for i, modality_acc in enumerate(self.val_modality_accs):
            self.log(f'val_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

        self.val_acc.reset()
        self.validation_step_outputs.clear()

    # ----------------- Test -----------------
    def test_step(self, batch, batch_idx):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.test_acc(evidences_a, target)

        for i, modality_acc in enumerate(self.test_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        alphas = evidences_a + 1
        denom = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denom
        entropy = self.num_classes / denom
        aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denom + 1)), dim=-1)

        self.test_step_outputs.append({
            'loss': loss.detach(),
            'entropy': entropy.detach(),
            'aleatoric': aleatoric.detach()
        })

    def on_test_epoch_end(self):
        entropies = torch.cat([x['entropy'] for x in self.test_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.test_step_outputs])

        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', entropies.mean())
        self.log('test_ale', aleatorics.mean())

        for i, modality_acc in enumerate(self.test_modality_accs):
            self.log(f'test_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

        self.test_acc.reset()
        self.test_step_outputs.clear()

    # ----------------- Optimizer -----------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class DisentangledEvidentialProbeModule(pl.LightningModule):
    """
    N-modal disentangled-only evidential probe.
    - Uses ONLY specific/private embeddings from the backbone.
    - Builds one evidential head per modality (no shared head).
    - Aggregation over the N specific streams via 'cml' or 'avg'.
    """
    def __init__(self, backbone, num_classes, input_dim, hidden_dim=(32), lr=1e-4, dropout=0.3,
                 annealing_start=20, optimizer=torch.optim.AdamW, freeze_backbone=True, aggregation='cml'):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        if not hasattr(self.backbone, 'N'):
            raise ValueError("backbone must expose attribute 'N' (number of modalities).")
        self.N = int(self.backbone.N)

        self.num_classes = num_classes
        self.lr = lr
        self.optimizer = optimizer
        self.annealing_start = annealing_start

        # N-agnostic aggregation (assumed to handle arbitrary #views)
        agg_map = {'cml': get_cml_fusion, 'avg': get_avg_fusion}
        if aggregation not in agg_map:
            raise ValueError(f"aggregation must be one of {list(agg_map.keys())}")
        self.agg = agg_map[aggregation]

        # One evidential head per modality (NO shared head here)
        self.spec_heads = nn.ModuleList([
            EvidentialNN(dropout=dropout, output_dims=num_classes, layers=(input_dim, *hidden_dim))
            for _ in range(self.N)
        ])

        # Metrics (overall + per-modality)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.num_views = self.N  # only specifics
        self.train_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                  for _ in range(self.num_views)])
        self.val_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                for _ in range(self.num_views)])
        self.test_modality_accs = nn.ModuleList([Accuracy(task='multiclass', num_classes=num_classes)
                                                 for _ in range(self.num_views)])

        # Loss that supports arbitrary num_views
        self.criterion = AvgTrustedLoss(num_views=self.num_views, annealing_start=annealing_start)

        if freeze_backbone:
            self.freeze_backbone()

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def get_embedding(self, batch):
        """
        Returns normalized embeddings:
          Zc: (B, N*input_dim)
          Zp_list: list of length N with tensors (B, input_dim)
        Accepts batch as [x0, ..., x_{N-1}, y] or [x0, ..., x_{N-1}]
        """
        xs = batch
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            last = batch[-1]
            if torch.is_tensor(last) and last.dtype in (torch.int64, torch.int32) and last.ndim >= 1:
                xs = batch[:-1]
        _, Zp_list = self.backbone.get_embedding(xs)  # expects (concat_shared, list_of_specific)
        return Zp_list

    def forward(self, x):
        """
        Returns a list of evidences, one per modality:
          [E_0, E_1, ..., E_{N-1}], each (B, num_classes)
        """
        Zp_list = self.get_embedding(x)
        evidences = [self.spec_heads[i](Zp_list[i]) for i in range(self.N)]
        return evidences

    def shared_step(self, x):
        labels = x[-1]
        evidences_list = self(x)                           # list length N
        evidences = torch.stack(evidences_list, dim=1)     # (B, N, C)
        evidences_a = self.agg(evidences)                  # (B, C)
        loss = self.criterion(evidences, labels, evidences_a)
        return loss, evidences_a, labels, evidences

    # ----------------- Training -----------------
    def training_step(self, batch, batch_idx=None):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self.train_acc(evidences_a, target)
        self.log('train_acc_step', acc, on_step=False, on_epoch=True, prog_bar=True)

        for i, modality_acc in enumerate(self.train_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)
            self.log(f'train_acc_modality_{i}_step', modality_acc.compute(), prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.criterion.annealing_step += 1
        for i, modality_acc in enumerate(self.train_modality_accs):
            self.log(f'train_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

    # ----------------- Validation -----------------
    def validation_step(self, batch, batch_idx):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.val_acc(evidences_a, target)

        for i, modality_acc in enumerate(self.val_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        alphas = evidences_a + 1
        denom = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denom
        entropy = self.num_classes / denom
        aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denom + 1)), dim=-1)

        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'entropy': entropy.detach(),
            'aleatoric': aleatoric.detach()
        })

    def on_validation_epoch_end(self):
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        entropies = torch.cat([x['entropy'] for x in self.validation_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.validation_step_outputs])

        self.log('val_loss', losses.mean(), on_step=False, prog_bar=True)
        self.log('val_entropy', entropies.mean(), on_step=False, prog_bar=True)
        self.log('val_sigma', aleatorics.mean(), on_step=False, prog_bar=True)
        self.log('val_acc', self.val_acc.compute(), on_step=False, prog_bar=True)

        for i, modality_acc in enumerate(self.val_modality_accs):
            self.log(f'val_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

        self.val_acc.reset()
        self.validation_step_outputs.clear()

    # ----------------- Test -----------------
    def test_step(self, batch, batch_idx):
        loss, evidences_a, target, evidences = self.shared_step(batch)
        self.test_acc(evidences_a, target)

        for i, modality_acc in enumerate(self.test_modality_accs):
            preds = torch.argmax(evidences[:, i, :], dim=1)
            modality_acc.update(preds, target)

        alphas = evidences_a + 1
        denom = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / denom
        entropy = self.num_classes / denom
        aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(denom + 1)), dim=-1)

        self.test_step_outputs.append({
            'loss': loss.detach(),
            'entropy': entropy.detach(),
            'aleatoric': aleatoric.detach()
        })

    def on_test_epoch_end(self):
        entropies = torch.cat([x['entropy'] for x in self.test_step_outputs])
        aleatorics = torch.cat([x['aleatoric'] for x in self.test_step_outputs])

        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', entropies.mean())
        self.log('test_ale', aleatorics.mean())

        for i, modality_acc in enumerate(self.test_modality_accs):
            self.log(f'test_acc_modality_{i}', modality_acc.compute(), on_step=False, prog_bar=False)
            modality_acc.reset()

        self.test_acc.reset()
        self.test_step_outputs.clear()

    # ----------------- Optimizer -----------------
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

