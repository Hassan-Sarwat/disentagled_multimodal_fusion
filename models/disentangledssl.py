import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import utils as utils
from losses import SupConLoss, ortho_loss
from utils import ExponentialScheduler
from classifiers import Linear, IdentityEncoder, MLP, ProbabilisticEncoder

from utils import augment_data 

torch.set_float32_matmul_precision('high')



class DisentangledSSL(pl.LightningModule):
    def __init__(self, feature_encoders=None, output_dim=[100,100], dropout=0.,  a=1,
                optimizer = torch.optim.Adam,  hidden_dim=512, embed_dim=100, 
                distribution='vmf',vmfkappa=1, lr=1e-4,lmd_start_value=0, 
                lmd_end_value=0, lmd_n_iterations=8000, lmd_start_iteration=0,
                ortho_norm=True, condzs=True,  usezsx=False, initialization='xavier', epochs=50):
        super(DisentangledSSL, self).__init__()

        self.optimizer = optimizer
        self.num_epochs = epochs
        x1_dim = output_dim[0]
        x2_dim = output_dim[1]
        self.N = 2
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        if feature_encoders is not None:
            self.feature_encoders = nn.ModuleList([i[0](**i[1]) for i in feature_encoders])
        else:
            self.feature_encoders = nn.ModuleList([IdentityEncoder() for _ in range(len(output_dim))])
        self.lr = lr
        self.ortho_norm = ortho_norm
        self.condzs = condzs
        self.usezsx = usezsx
        self.vmfkappa = vmfkappa
        self.iterations = 0
        if lmd_end_value > 0:
            self.lmd_scheduler = ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                    n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.a = a

        self.encoder_x1s = Linear(layers = (x1_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)
        self.encoder_x2s = Linear(layers = (x2_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)

        self.phead1 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)
        self.phead2 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)

        if self.condzs:
            self.encoder_x1 = Linear(layers = (x1_dim+embed_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)
            self.encoder_x2 = Linear(layers = (x2_dim+embed_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)
        else:
            self.encoder_x1 = Linear(layers = (x1_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)
            self.encoder_x2 = Linear(layers = (x2_dim, hidden_dim, hidden_dim), output_dims = embed_dim, initialization = initialization, dropout=0)

        self.critic = SupConLoss()
        
    
    def get_embedding(self, x):
        x1 = x[0].float()
        x2 = x[1].float()
        x1 = self.feature_encoders[0](x1)
        x2 = self.feature_encoders[1](x2)
        zsx1 = self.encoder_x1s(x1)
        zsx2 = self.encoder_x2s(x2)
        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z2x2 = self.encoder_x2(x2)
        return torch.cat([zsx1, zsx2], dim=1), [z1x1, z2x2]

    def forward(self, x1,x2,v1,v2):
        

        x1 = self.feature_encoders[0](x1)
        v1 = self.feature_encoders[0](v1)
        x2 = self.feature_encoders[1](x2)
        v2 = self.feature_encoders[1](v2)
        
        e1 = self.encoder_x1s(x1)
        e2 = self.encoder_x2s(x2)
        e1_v = self.encoder_x1s(v1)
        e2_v = self.encoder_x2s(v2)

        p_zs1_given_x1, mu1 = self.phead1(e1)
        p_zs2_given_x2, mu2 = self.phead2(e2)
        p_zsv1_given_v1, mu1_v = self.phead1(e1_v)
        p_zsv2_given_v2, mu2_v = self.phead2(e2_v)

        zs1 = p_zs1_given_x1.rsample()
        zs2 = p_zs2_given_x2.rsample()
        zsv1 = p_zsv1_given_v1.rsample()
        zsv2 = p_zsv2_given_v2.rsample()


        concat_embed = torch.cat([zs1.unsqueeze(dim=1), zs2.unsqueeze(dim=1)], dim=1)
        concat_embed_v = torch.cat([zsv1.unsqueeze(dim=1), zsv2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)
        joint_loss_v, loss_x_v, loss_y_v = self.critic(concat_embed_v)
        joint_loss = 0.5 * (joint_loss + joint_loss_v)
        loss_x = 0.5 * (loss_x + loss_x_v)
        loss_y = 0.5 * (loss_y + loss_y_v)
        loss_shared = joint_loss


        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, e1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, e1_v], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, e2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, e2_v], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        
        if self.usezsx:
            zjointx1 = torch.cat([z1x1, e1], dim=1)
            zjointx2 = torch.cat([z2x2, e2], dim=1)
            zjointxv1 = torch.cat([z1xv1, e1_v], dim=1)
            zjointxv2 = torch.cat([z2xv2, e2_v], dim=1)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, e1, norm=self.ortho_norm) + ortho_loss(z2x2, e2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, e1_v, norm=self.ortho_norm) + ortho_loss(z2xv2, e2_v, norm=self.ortho_norm))
        
        loss = 2 * loss_shared/(1+self.a) + self.a * loss_specific/(1+self.a) + lmd * loss_ortho

        return loss, {'loss': loss.item(), 'shared': loss_shared.item(), 'clip': joint_loss.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(),
                       'specific': loss_specific.item(), 'ortho': loss_ortho.item(), 'lmd': lmd}#, 'beta': beta,

    def training_step(self, batch, batch_idx):
        x1, x2, v1, v2 = self.shared_step(batch)
        loss, train_logs = self(x1, x2, v1, v2)
        self.iterations += 1
        self.log('train_loss', train_logs['loss'],  on_epoch=True, prog_bar=True)
        self.log('shared', train_logs['shared'],  on_epoch=True, prog_bar=True)
        self.log('clip', train_logs['clip'], on_epoch=True, prog_bar=True)
        self.log('loss_x', train_logs['loss_x'], on_epoch=True, prog_bar=True)
        self.log('loss_y', train_logs['loss_y'], on_epoch=True, prog_bar=True)
        self.log('specific', train_logs['specific'], on_epoch=True, prog_bar=True)
        self.log('ortho', train_logs['ortho'], on_epoch=True, prog_bar=True)
        self.log('lmd', train_logs['lmd'], on_epoch=True, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x1 = batch[0].float().cuda()
        x2 = batch[1].float().cuda()
        v1 = augment_data(x1)
        v2 = augment_data(x2)   
        return x1, x2, v1, v2

    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0, last_epoch=-1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step' if needed
                'monitor': 'train_loss',  # optional, used for ReduceLROnPlateau
            }
        }
    


