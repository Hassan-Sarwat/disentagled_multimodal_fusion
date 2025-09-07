import pandas as pd
import pytorch_lightning as pl
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe
from models.dmvae import DMVAE
import models.baselines as baselines
from models.evidential_probe import EvidentialProbeModule
from dataset import make_loaders_simple_plus
from classifiers import IdentityEncoder

COMMON_MED = dict(
    n_samples=10000,
    d_signal=16,
    d_spurious=16,          # ↓ distractors
    alpha_shared=0.7,
    beta_specific=0.6,
    class_sep_shared=1.1,   # ↑ separation a bit
    class_sep_private=0.9,  # ↑ private separation a bit
    noise_std=0.7,          # ↓ noise
    hetero_noise=True,
    hetero_scale=0.4,       # ↓ heteroscedasticity
    nonlinear_shared=True,
    nonlinear_specific=False, # turn off private nonlinearity
    conflict_frac=0.4,      # fewer conflict classes
    conflict_strength=0.7,  # softer conflict
)

def make_dep_loader_med(dep_percent, seed=7, **overrides):
    rho = dep_percent / 100.0
    return make_loaders_simple_plus(
        seed=seed,
        rho=rho,
        shared_class_frac=rho,     # keep class sharing tied to dependence
        **{**COMMON_MED, **overrides}
    )


def train_dmvae(train_loader, seed, dep, a=1e-5,hidden_dim=512, embed_dim=16,lr=1e-3, output_dim=[32,32], num_epochs=100):
    dmvae = DMVAE(
    output_dim=output_dim,     # match X1/X2 dims
    hidden_dim=hidden_dim,          # 256 if inputs ~100-D
    embed_dim=embed_dim,             # match ds=8
    lr=lr,                 # AdamW; keep weight_decay=1e-4 in optimizer
    a=a,                   # with KL warm-up 0→1 over first 30% epochs
    num_epochs=num_epochs
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        log_every_n_steps=20, # For speed
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(dmvae, train_dataloaders=train_loader)
    dmvae_name = f'checkpoints/dmvae_seed{seed}_dep{dep}_a{1e-5}_hd{hidden_dim}_lr{lr}_ed{embed_dim}.ckpt'
    trainer.save_checkpoint(dmvae_name)
    return dmvae

def train_dmvae_fusion(dmvae, train_loader, test_loader, seed, dep, annealing_start=10, lr=3e-4,num_classes=3,
                        num_epochs=50,dropout=0.1, aggregation='cml',input_dim=16, hidden_dim=(128,)):
    
    model_name = f'checkpoints/dmvae_fusion_seed{seed}_dep{dep}_agg{aggregation}_hd{hidden_dim}_lr{lr}.ckpt'
    dmvae_fusion = EvidentialProbeModule(backbone=dmvae,num_classes=num_classes, input_dim=input_dim, aggregation=aggregation, dropout=dropout,
                annealing_start = annealing_start, lr=lr, hidden_dim=hidden_dim, freeze_backbone=True, fused=0)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        log_every_n_steps=20, # For speed
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Lightning will call model.training_step & model.validation_step
    trainer.fit(dmvae_fusion, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.save_checkpoint(model_name)
    return dmvae_fusion

def train_latefusion(train_loader, test_loader, seed, dep, aggregation, annealing_start=10, dropout=0.1, output_dims=[32,32],num_classes=3,
                     hidden_dim=(128,), lr=3e-4, classifiers = [(IdentityEncoder, {}), (IdentityEncoder, {})], num_epochs=50):
    
    fusion = baselines.LateFusion(classifiers, output_dims, num_classes = num_classes, dropout=dropout, 
                            aggregation=aggregation, annealing_start = annealing_start, lr=lr, hidden_dim=hidden_dim, fused=0)
    model_name = f'checkpoints/late_fusion_seed{seed}_dep{dep}_agg{aggregation}_hd{hidden_dim}_lr{lr}.ckpt'
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        log_every_n_steps=20, # For speed
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Lightning will call model.training_step & model.validation_step
    trainer.fit(fusion, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.save_checkpoint(model_name)
    return fusion




rows = {}
for seed in [0,1,2,3,4,5]:
    rows[seed] = {}
    for dep in [0,25,50,75,100]:
        pl.seed_everything(seed)
        rows[seed][dep] = {}
        ds, train_loader, test_loader = make_dep_loader_med(dep, seed=seed)
        dmvae = train_dmvae(train_loader, seed, dep, a=1e-5,)

        dmvae_fusion = train_dmvae_fusion(dmvae,train_loader, test_loader, seed, dep)
        rows[seed][dep]['dmvae_cml'] = evaluate_subjective_model_with_shared(dmvae_fusion, test_loader)
        
        cml_fusion = train_latefusion(train_loader, test_loader, seed, dep, aggregation='cml')
        rows[seed][dep]['cml'] = evaluate_subjective_model(cml_fusion, test_loader)

        avg_fusion = train_latefusion(train_loader, test_loader, seed, dep, aggregation='avg')
        rows[seed][dep]['avg'] = evaluate_subjective_model(avg_fusion, test_loader)



df  = build_metrics_dataframe(rows)
df['seed'] = df['seed'].astype(int)
df['dep'] = df['dep'].astype(float)
df_main = df[['seed','dep','model','view_0_evidence_mean','view_1_evidence_mean', 'shared_evidence_mean', 'fused_evidence_mean',
                  'view_0_aleatoric_mean', 'view_1_aleatoric_mean','shared_aleatoric_mean','fused_aleatoric_mean',
                  'view_0_epistemic_mean','view_1_epistemic_mean','shared_epistemic_mean','fused_epistemic_mean',
                 'view_0_accuracy',  'view_1_accuracy', 'shared_accuracy', 'fused_accuracy']]

df_grouped = df.groupby(['dep','model']).mean().reset_index()
df_grouped.sort_values(by=['dep','model'],inplace=True)
df_main_grouped = df_main.groupby(['dep','model']).mean().reset_index()
df_main_grouped.sort_values(by=['dep','model'],inplace=True)
with  pd.ExcelWriter('logs/synthetic_dataset.xlsx') as writer:
    df_main_grouped.to_excel(writer, sheet_name='main_grouped',index=False)
    df.to_excel(writer, sheet_name='all_results',index=False)
    df_grouped.to_excel(writer,sheet_name='grouped_results',index=False)

        
