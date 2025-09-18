import pandas as pd
import pytorch_lightning as pl
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe
from models.dmvae import DMVAE
import models.baselines as baselines
from models.evidential_probe import EvidentialProbeModule
from dataset import make_loaders_simple_plus
from classifiers import IdentityEncoder
import yaml
from pathlib import Path

CFG_PATH = Path("configs/synthetic_config.yaml")
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Helpers with sensible fallbacks (preserve original behavior if a key is missing)
def C(path, default=None):
    """Dot-path getter with default, e.g., C('data.common_med.alpha_shared', 0.7)"""
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

COMMON_MED = dict(
    n_samples=C("data.common_med.n_samples", 10000),
    d_signal=C("data.common_med.d_signal", 16),
    d_spurious=C("data.common_med.d_spurious", 16),
    alpha_shared=C("data.common_med.alpha_shared", 0.7),
    beta_specific=C("data.common_med.beta_specific", 0.6),
    class_sep_shared=C("data.common_med.class_sep_shared", 1.1),
    class_sep_private=C("data.common_med.class_sep_private", 0.9),
    noise_std=C("data.common_med.noise_std", 0.7),
    hetero_noise=C("data.common_med.hetero_noise", True),
    hetero_scale=C("data.common_med.hetero_scale", 0.4),
    nonlinear_shared=C("data.common_med.nonlinear_shared", True),
    nonlinear_specific=C("data.common_med.nonlinear_specific", False),
    conflict_frac=C("data.common_med.conflict_frac", 0.4),
    conflict_strength=C("data.common_med.conflict_strength", 0.7),
)

def make_dep_loader_med(dep_percent, seed=7, **overrides):
    rho = dep_percent / 100.0
    return make_loaders_simple_plus(
        seed=seed,
        rho=rho,
        shared_class_frac=rho,     # keep class sharing tied to dependence
        **{**COMMON_MED, **overrides}
    )


def train_dmvae(train_loader, seed, dep, a=C("dmvae.a", 1e-5),hidden_dim=C("dmvae.hidden_dim", 512), 
                embed_dim=C("dmvae.embed_dim", 16),lr=C("dmvae.lr", 1e-3), output_dim=C("dmvae.output_dim", [32, 32]), 
                num_epochs=C("dmvae.num_epochs", 100)):
    
    dmvae = DMVAE(output_dim=output_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, 
                  lr=lr, a=a, num_epochs=num_epochs)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        log_every_n_steps=20, # For speed
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(dmvae, train_dataloaders=train_loader)
    dmvae_name = f'checkpoints/dmvae_seed{seed}_dep{dep}_a{a}_hd{hidden_dim}_lr{lr}_ed{embed_dim}.ckpt'
    trainer.save_checkpoint(dmvae_name)
    return dmvae

def train_dmvae_fusion(dmvae, train_loader, test_loader, seed, dep, annealing_start=C("dmvae_fusion.annealing_start", 10), 
                       lr=C("dmvae_fusion.lr", 3e-4),num_classes=C("dmvae_fusion.num_classes", 3), num_epochs=C("dmvae_fusion.num_epochs", 50),
                       dropout=C("dmvae_fusion.dropout", 0.1), aggregation=C("dmvae_fusion.aggregation", "cml"),input_dim=C("dmvae_fusion.input_dim", 16), 
                       hidden_dim=tuple(C("dmvae_fusion.hidden_dim", (128,)))):
    
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

CLASSIFIER_REGISTRY = {
    "IdentityEncoder": IdentityEncoder,
    # Add more here if you want to select different encoders from YAML later
}

def _build_classifiers_from_cfg():
    cls_cfg_list = C("latefusion.classifiers", None)
    if not cls_cfg_list:
        # default to your original two IdentityEncoders
        return [(IdentityEncoder, {}), (IdentityEncoder, {})]
    out = []
    for item in cls_cfg_list:
        name = item.get("name")
        kwargs = item.get("kwargs", {}) or {}
        if name not in CLASSIFIER_REGISTRY:
            raise ValueError(f"Unknown classifier name in config: {name}")
        out.append((CLASSIFIER_REGISTRY[name], kwargs))
    return out

def train_latefusion(train_loader, test_loader, seed, dep, aggregation, annealing_start=C("latefusion.annealing_start", 10), dropout=C("latefusion.dropout", 0.1), 
                     output_dims=C("latefusion.output_dims", [32, 32]),num_classes=C("latefusion.num_classes", 3), hidden_dim=tuple(C("latefusion.hidden_dim", (128,))), 
                     lr=C("latefusion.lr", 3e-4), classifiers = _build_classifiers_from_cfg(), num_epochs=C("latefusion.num_epochs", 50)):
    
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



seeds = C("experiment.seeds", [0, 1, 2, 3, 4, 5])
deps = C("experiment.deps", [0, 25, 50, 75, 100])

rows = {}
for seed in seeds:
    rows[seed] = {}
    for dep in deps:
        pl.seed_everything(seed)
        rows[seed][dep] = {}
        ds, train_loader, test_loader = make_dep_loader_med(dep, seed=seed)
        dmvae = train_dmvae(
            train_loader=train_loader,
            seed=seed,
            dep=dep,
            a=C("dmvae.a", 1e-5),
            hidden_dim=C("dmvae.hidden_dim", 512),
            embed_dim=C("dmvae.embed_dim", 16),
            lr=C("dmvae.lr", 1e-3),
            output_dim=C("dmvae.output_dim", [32, 32]),
            num_epochs=C("dmvae.num_epochs", 100),
        )

        dmvae_fusion = train_dmvae_fusion(
            dmvae=dmvae,
            train_loader=train_loader,
            test_loader=test_loader,
            seed=seed,
            dep=dep,
            annealing_start=C("dmvae_fusion.annealing_start", 10),
            lr=C("dmvae_fusion.lr", 3e-4),
            num_classes=C("dmvae_fusion.num_classes", 3),
            num_epochs=C("dmvae_fusion.num_epochs", 50),
            dropout=C("dmvae_fusion.dropout", 0.1),
            aggregation=C("dmvae_fusion.aggregation", "cml"),
            input_dim=C("dmvae_fusion.input_dim", 16),
            hidden_dim=tuple(C("dmvae_fusion.hidden_dim", (128,))),
        )
        rows[seed][dep]['dmvae_cml'] = evaluate_subjective_model_with_shared(dmvae_fusion, test_loader)
        
        cml_fusion = train_latefusion(
            train_loader=train_loader,
            test_loader=test_loader,
            seed=seed,
            dep=dep,
            aggregation="cml",  # keep explicit; filename uses this
            annealing_start=C("latefusion.annealing_start", 10),
            dropout=C("latefusion.dropout", 0.1),
            output_dims=C("latefusion.output_dims", [32, 32]),
            num_classes=C("latefusion.num_classes", 3),
            hidden_dim=tuple(C("latefusion.hidden_dim", (128,))),
            lr=C("latefusion.lr", 3e-4),
            classifiers=None,  # will use YAML if provided
            num_epochs=C("latefusion.num_epochs", 50),
        )
        rows[seed][dep]["cml"] = evaluate_subjective_model(cml_fusion, test_loader)

        avg_fusion = train_latefusion(
            train_loader=train_loader,
            test_loader=test_loader,
            seed=seed,
            dep=dep,
            aggregation="avg",  # keep explicit; filename uses this
            annealing_start=C("latefusion.annealing_start", 10),
            dropout=C("latefusion.dropout", 0.1),
            output_dims=C("latefusion.output_dims", [32, 32]),
            num_classes=C("latefusion.num_classes", 3),
            hidden_dim=tuple(C("latefusion.hidden_dim", (128,))),
            lr=C("latefusion.lr", 3e-4),
            classifiers=None,  # will use YAML if provided
            num_epochs=C("latefusion.num_epochs", 50),
        )
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

        
