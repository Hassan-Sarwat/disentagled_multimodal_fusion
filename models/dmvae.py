import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import utils as utils
from classifiers import Linear, IdentityEncoder
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')




class DMVAE(pl.LightningModule):
    """
    N-modal DMVAE (N >= 2).
    - Each modality i has:
        encoder_i: x_i -> [mu_s_i, logvar_s_i, mu_p_i, logvar_p_i]  (dims = 4*embed_dim)
        decoder_i: [z_p_i, z_s] -> x_i_hat
    - Shared posterior uses Product-of-Experts (PoE) over all modalities’ shared Gaussians.
    - Recon losses:
        * Joint/self: for each i, decode with z_s from PoE (x_i ← (z_p_i, z_s))
        * Cross: for each ordered pair (i, j) with i != j, decode x_i using z_s_j (x_i ← (z_p_i, z_s_j))
    - KL losses:
        * Private: sum_i KL(q(z_p_i|x_i) || p(z_p_i))
        * Shared (joint): N * KL(q_PoE(z_s|x) || p(z_s))   # scales like your 2× term for N=2
        * Shared (cross side): sum_j KL(q(z_s|x_j) || p(z_s))   # like the paper’s second group
    """
    def __init__(
        self, feature_encoders=None, output_dim=[100, 100], dropout=0., a=1.0,
        optimizer=torch.optim.Adam, hidden_dim=512, embed_dim=100, lr=1e-4, initialization='xavier',
        num_epochs=50,poe_temperature=1.5,cross_weight = 1.0,lambda_per_modality = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['feature_encoders'])  # neat Lightning logging
        self.num_epochs = num_epochs
        self.optimizer_cls = optimizer
        self.lr = lr
        self.a = a
        # ----- modalities meta -----
        assert isinstance(output_dim, (list, tuple)) and len(output_dim) >= 2, \
            "output_dim must be a list of per-modality input dims (N >= 2)."
        self.N = len(output_dim)
        self.x_dims = list(output_dim)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim  # size of each of shared/private vectors
        self.poe_temperature = poe_temperature
        self.cross_weight = cross_weight
        self.lambda_per_modality = lambda_per_modality or [1.0]*self.N

        # ----- feature encoders (pre-encoders before our MLP encoders) -----
        if feature_encoders is not None:
            assert len(feature_encoders) == self.N, "feature_encoders length must equal number of modalities."
            self.feature_encoders = nn.ModuleList([ctor(**kwargs) for ctor, kwargs in feature_encoders])
        else:
            self.feature_encoders = nn.ModuleList([IdentityEncoder() for _ in range(self.N)])

        # ----- inference encoders: x_i -> 4*embed_dim -----
        # chunk order: [mu_s_i, logvar_s_i, mu_p_i, logvar_p_i]
        self.encoders = nn.ModuleList([
            Linear(layers=(self.x_dims[i], hidden_dim, hidden_dim),
                   output_dims=4*embed_dim, initialization=initialization, dropout=dropout)
            for i in range(self.N)
        ])

        # ----- decoders: concat([z_p_i, z_s]) -> x_i_dim -----
        self.decoders = nn.ModuleList([
            Linear(layers=(2*embed_dim, hidden_dim, hidden_dim),
                   output_dims=self.x_dims[i], initialization=initialization, dropout=dropout)
            for i in range(self.N)
        ])

    # ---------- utils ----------
    @staticmethod
    def _chunk_shared_private(four_e):
        # returns (mu_s, logvar_s, mu_p, logvar_p)
        mu_s, logvar_s, mu_p, logvar_p = four_e.chunk(4, dim=1)
        return mu_s, logvar_s, mu_p, logvar_p

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu, logvar):
        # KL( N(mu, sigma^2) || N(0,1) ) summed over dims, averaged over batch later.
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)

    @staticmethod
    def product_of_experts(mu_list, logvar_list, temperature: float = 1.0, include_prior: bool = True):
        """
        Gaussian PoE with optional temperature and prior-as-expert.
        q(z|x) ∝ p(z) ∏_i q_i(z|x_i).  Temperature T>1 softens experts (less confident).
        """
        mus     = torch.stack(mu_list, dim=0)       # (K,B,D)
        logvars = torch.stack(logvar_list, dim=0)   # (K,B,D)

        if include_prior:
            prior_mu     = torch.zeros_like(mus[0])       # N(0, I)
            prior_logvar = torch.zeros_like(logvars[0])   # log(1) = 0
            mus     = torch.cat([mus,     prior_mu.unsqueeze(0)], dim=0)
            logvars = torch.cat([logvars, prior_logvar.unsqueeze(0)], dim=0)

        # tempered precisions: raising each expert to power 1/T ≈ divide precision by T
        precisions = torch.exp(-logvars) / max(temperature, 1e-8)
        precision_sum = precisions.sum(dim=0) + 1e-8
        var = 1.0 / precision_sum
        mu  = var * (precisions * mus).sum(dim=0)
        logvar = torch.log(var)
        return mu, logvar

    # ---------- public helpers ----------
    @torch.no_grad()
    def get_embedding(self, x_list, return_poe: bool = True):
        feats = [self.feature_encoders[i](x_list[i]) for i in range(self.N)]
        stats = [self._chunk_shared_private(self.encoders[i](feats[i])) for i in range(self.N)]
        mu_s_all = [s[0] for s in stats]
        mu_p_all = [s[2] for s in stats]
        if return_poe:
            mu_s_poe, _ = self.product_of_experts(mu_s_all, [s[1] for s in stats], temperature=self.poe_temperature, include_prior=True)
            return mu_s_poe, mu_p_all
        else:
            return torch.cat(mu_s_all, dim=1), mu_p_all

    # ---------- core forward ----------
    def forward(self, x_list):
        """
        x_list: list of N tensors, each (B, S_i)
        Returns: loss, logs (dict)
        """
        B = x_list[0].shape[0]

        # 1) feature pre-encoders
        feats = [self.feature_encoders[i](x_list[i]) for i in range(self.N)]

        # 2) encode -> stats
        stats = [self._chunk_shared_private(self.encoders[i](feats[i])) for i in range(self.N)]
        # --- after stats and before sampling ---
        mu_s_list   = [s[0] for s in stats]
        logv_s_list = [s[1] for s in stats]
        mu_p_list   = [s[2] for s in stats]
        logv_p_list = [s[3] for s in stats]

        # 3) sample z_p_i, z_s_i (unimodal), and z_s (PoE)
        z_p_list = [self.reparameterize(mu_p_list[i], logv_p_list[i]) for i in range(self.N)]
        z_s_uni  = [self.reparameterize(mu_s_list[i], logv_s_list[i]) for i in range(self.N)]
        mu_s_poe, logv_s_poe = self.product_of_experts(mu_s_list, logv_s_list, temperature=1.5, include_prior=True)
        z_s = self.reparameterize(mu_s_poe, logv_s_poe)


        lam = self.lambda_per_modality
        x_recon_joint = [ self.decoders[i](torch.cat([z_p_list[i], z_s], dim=1)) for i in range(self.N) ]
        loss_recon_joint = sum(lam[i]*F.mse_loss(x_recon_joint[i], feats[i]) for i in range(self.N))

        loss_recon_cross, count_pairs = 0.0, 0
        x_recon_cross = [[None]*self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if i == j: continue
                x_hat = self.decoders[i](torch.cat([z_p_list[i], z_s_uni[j]], dim=1))
                x_recon_cross[i][j] = x_hat
                loss_recon_cross += lam[i]*F.mse_loss(x_hat, feats[i])
                count_pairs += 1
        if count_pairs > 0:
            loss_recon_cross = (loss_recon_cross / count_pairs) * self.cross_weight

        # KLs
        kl_private = torch.stack([self.kl_divergence(mu_p_list[i], logv_p_list[i]) for i in range(self.N)], dim=1).sum(dim=1).mean()
        kl_shared_poe = self.kl_divergence(mu_s_poe, logv_s_poe).mean()
        kl_shared_uni = torch.stack([self.kl_divergence(mu_s_list[i], logv_s_list[i]) for i in range(self.N)], dim=1).sum(dim=1).mean()

        loss_joint = loss_recon_joint + self.a * (kl_private + self.N * kl_shared_poe)
        loss_cross = loss_recon_cross + self.a * (kl_shared_uni)
        loss = loss_joint + loss_cross

        logs = {
            'loss': loss.detach(),
            'loss_joint_recon': float(loss_recon_joint),
            'loss_cross_recon': float(loss_recon_cross) if count_pairs > 0 else 0.0,
            'kl_private': float(kl_private),
            'kl_shared_poe': float(kl_shared_poe),
            'kl_shared_uni_sum': float(kl_shared_uni),
            'a': float(self.a),
            'N': self.N
        }
        return loss, logs

    # ---------- Lightning plumbing ----------
    def training_step(self, batch, batch_idx):
        # batch is a list: [x0, x1, ..., x_{N-1}, y]; we ignore the label here
        xs = [b.float() for b in batch[:-1]]
        loss, logs = self(xs)
        # progress bar / logs
        self.log('train/loss', logs['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loss_joint_recon', logs['loss_joint_recon'], on_epoch=True, prog_bar=True)
        self.log('train/loss_cross_recon', logs['loss_cross_recon'], on_epoch=True, prog_bar=True)
        self.log('train/kl_private', logs['kl_private'], on_epoch=True, prog_bar=True)
        self.log('train/kl_shared_poe', logs['kl_shared_poe'], on_epoch=True, prog_bar=True)
        self.log('train/kl_shared_uni_sum', logs['kl_shared_uni_sum'], on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = self.optimizer_cls(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_epochs, eta_min=0, last_epoch=-1)
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sch, 'interval': 'epoch', 'monitor': 'train/loss'}
        }



