import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from pprint import pprint
from pathlib import Path
import yaml
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, Subset, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import json
import models.baselines as baselines
from datasets.dataset import CUB, Caltech, HandWritten, PIE, Scene
from models.classifiers import IdentityEncoder
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe_datasets
from  models.dmvae import DMVAE
from models.evidential_probe  import EvidentialProbeModule, DisentangledEvidentialProbeModule
from functools import partial

CFG_PATH = Path("configs/config.yaml")
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

def C(path, default=None):
    """Dot-path getter with default: C('probes.dropout_p', 0.1)"""
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _get_dataset(dataset_name):
    if dataset_name == "CUB":
        return CUB()
    elif dataset_name == "CalTech":
        return Caltech()
    elif dataset_name == "HandWritten":
        return HandWritten()
    elif dataset_name == "PIE":
        return PIE()
    elif dataset_name == "Scene":
        return Scene()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def _split_indices(n, train_frac):
    idx = np.arange(n)
    np.random.shuffle(idx)  # numpy rng is seeded by pl.seed_everything(seed)
    n_train = int(train_frac * n)
    return idx[:n_train], idx[n_train:]


def get_normal_data(dataset_name):
    batch_size = C("dataloader.batch_size", 100)
    num_workers = C("dataloader.num_workers", 0)
    train_frac = C("data.split.train_frac", 0.8)

    dataset = _get_dataset(dataset_name)
    n = len(dataset)
    train_idx, test_idx = _split_indices(n, train_frac)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = dataset.num_classes
    num_views   = dataset.num_views
    dims        = list(np.squeeze(dataset.dims))
    return train_loader, test_loader, num_classes, num_views, dims

def get_conflict_data(dataset_name):
    batch_size = C("dataloader.batch_size", 100)
    num_workers = C("dataloader.num_workers", 0)
    train_frac = C("data.split.train_frac", 0.8)

    dataset = _get_dataset(dataset_name)
    n = len(dataset)
    train_idx, test_idx = _split_indices(n, train_frac)

    # Conflict/post-processing controls from YAML
    pp = C("data.conflict", {})
    dataset.postprocessing(
        test_idx,
        addNoise=pp.get("addNoise", False),
        sigma=pp.get("sigma", 0.5),
        ratio_noise=pp.get("ratio_noise", 0.0),
        addConflict=pp.get("addConflict", True),
        ratio_conflict=pp.get("ratio_conflict", 1.0),
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = dataset.num_classes
    num_views   = dataset.num_views
    dims        = list(np.squeeze(dataset.dims))
    return train_loader, test_loader, num_classes, num_views, dims


seeds = C("experiment.seeds", [0, 1, 2, 3, 4])
normal_datasets   = C("experiment.normal_datasets",   ['CUB', 'CalTech', 'HandWritten', 'PIE', 'Scene'])
conflict_datasets = C("experiment.conflict_datasets", ['CUB', 'CalTech', 'HandWritten', 'PIE', 'Scene'])

dataset_lr = C("optim.dataset_lr", {
    "CalTech": 0.0003, "Scene": 0.01, "CUB": 0.003, "HandWritten": 0.003, "PIE": 0.003
})

model_parameters = {
    "dropout_p":       C("probes.dropout_p", 0.1),
    "annealing_start": C("probes.annealing_start", 50),
    "model_epochs":    C("probes.model_epochs", 200),
    "model_hidden_dim": tuple(C("probes.model_hidden_dim", (128,))),
}

probe_input_dim = C("probes.input_dim", 200)

dmvae_kwargs = {
    "dropout":    C("dmvae.dropout", 0),
    "a":          C("dmvae.a", 1e-5),
    "hidden_dim": C("dmvae.hidden_dim", 512),
    "embed_dim":  C("dmvae.embed_dim", 200),
    "lr":         C("dmvae.lr", 1e-4),
    "num_epochs": C("dmvae.num_epochs", 100),
}

dataset_lr = C("optim.dataset_lr", {
    "CalTech": 0.0003, "Scene": 0.01, "CUB": 0.003, "HandWritten": 0.003, "PIE": 0.003
})

def build_factories(model_params, probe_input_dim, dmvae_kwargs):
    DMVAEFactory = partial(
        DMVAE,
        feature_encoders=model_params["classifiers"],
        output_dim=model_params["output_dims"],
        dropout=dmvae_kwargs["dropout"],
        a=dmvae_kwargs["a"],
        hidden_dim=dmvae_kwargs["hidden_dim"],
        embed_dim=dmvae_kwargs["embed_dim"],
        lr=dmvae_kwargs["lr"],
        num_epochs=dmvae_kwargs["num_epochs"],
    )
    ProbeFactory = partial(
        EvidentialProbeModule,
        num_classes=model_params["classes"],
        lr=model_params["lr"],
        annealing_start=model_params["annealing_start"],
        hidden_dim=model_params["model_hidden_dim"],
        dropout=model_params["dropout_p"],
        input_dim=probe_input_dim,
    )
    DisProbeFactory = partial(
        DisentangledEvidentialProbeModule,
        num_classes=model_params["classes"],
        lr=model_params["lr"],
        annealing_start=model_params["annealing_start"],
        hidden_dim=model_params["model_hidden_dim"],
        dropout=model_params["dropout_p"],
        input_dim=probe_input_dim,
    )
    LateFusionFactory = partial(
        baselines.LateFusion,
        model_params["classifiers"],
        model_params["output_dims"],
        model_params["classes"],
        dropout=model_params["dropout_p"],
        lr=model_params["lr"],
        annealing_start=model_params["annealing_start"],
        hidden_dim=model_params["model_hidden_dim"],
    )
    return DMVAEFactory, ProbeFactory, DisProbeFactory, LateFusionFactory



rows = {}
for seed in seeds:
    pl.seed_everything(seed)
    rows[seed] = {}
    ###  Normal Loop
    rows[seed]['Normal'] =  {}
    for dataset_name in normal_datasets:
        rows[seed]['Normal'][dataset_name] = {}
        train_loader, test_loader, num_classes, num_views, dims = get_normal_data(dataset_name)
        model_parameters["classes"] = num_classes
        model_parameters["output_dims"] = dims
        model_parameters['classifiers'] = [(IdentityEncoder, {}) for i in range(len(dims))]
        model_parameters['lr']  = dataset_lr[dataset_name]

        DMVAEFactory, ProbeFactory, DisProbeFactory, LateFusionFactory = build_factories(
            model_parameters, probe_input_dim, dmvae_kwargs
        )

        dmvae_model = DMVAEFactory()

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=dmvae_kwargs["num_epochs"],
            enable_progress_bar=True,
            log_every_n_steps=20,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        model_name = f'dmvae_dataset{dataset_name}_seed{seed}_a{1e-5}_normal'
        path = f'checkpoints/{model_name}.ckpt'
        trainer.save_checkpoint(path)

        dis_dmvae_model  = DisProbeFactory(dmvae_model)
        cml_dmvae_model  = ProbeFactory(dmvae_model, aggregation="cml")
        joint_dmvae_model = ProbeFactory(dmvae_model, aggregation="joint")
        
        dbf_fusion = LateFusionFactory(aggregation="dbf")
        cml_fusion = LateFusionFactory(aggregation="cml")
        avg_fusion = LateFusionFactory(aggregation="avg")

        models = [dis_dmvae_model, cml_dmvae_model, joint_dmvae_model, dbf_fusion, cml_fusion, avg_fusion]
        model_names = ['dmvae_dis','dmvae_cml','dmvae_joint','dbf_fusion','cml_fusion','avg_fusion']

        
        for model, name in zip(models, model_names):
            model_name = f'{name}_fusion_ds{dataset_name}_seed{seed}'
            print(f'Training model {model_name}')
            tags=[name,str(seed), dataset_name,'Normal']
            model_parameters.update({"dataset": dataset_name, "seed": seed,'aggregation':name ,'UQ':'Normal'})
            csv_logger = CSVLogger(
                save_dir="logs/",
                name=model_name,
            )

            gpus = 1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(
                accelerator="auto" if gpus else "cpu",
                devices=gpus if gpus else None,
                max_epochs=model_parameters['model_epochs'],
                log_every_n_steps=20, # For speed
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=csv_logger        # turn off TensorBoard for brevity
            )

            # Lightning will call model.training_step & model.validation_step
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            # Load best checkpoint after training
            # Test best model on the test set
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            path = f'checkpoints/{model_name}.ckpt'
            trainer.save_checkpoint(path)
            print(f'Model {model_name}')
            pprint(test_metrics)
            if name == 'dmvae_dis':
                rows[seed]['Normal'][dataset_name][name] = evaluate_subjective_model(model, test_loader)
            else:    
                rows[seed]['Normal'][dataset_name][name] = evaluate_subjective_model_with_shared(model, test_loader)
            rows[seed]['Normal'][dataset_name][name].update({'path':path})

    rows[seed]['Conflict'] = {}
    for dataset_name in conflict_datasets:
        rows[seed]['Conflict'][dataset_name] = {}
        train_loader, test_loader, num_classes, num_views, dims = get_conflict_data(dataset_name)
        model_parameters["classes"] = num_classes
        model_parameters["output_dims"] = dims
        model_parameters['classifiers'] = [(IdentityEncoder, {}) for _ in range(len(dims))]
        model_parameters['lr']  = dataset_lr[dataset_name]

        DMVAEFactory, ProbeFactory, DisProbeFactory, LateFusionFactory = build_factories(
            model_parameters, probe_input_dim, dmvae_kwargs
        )

        dmvae_model = DMVAEFactory()
        

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=100,
            enable_progress_bar=True,
            log_every_n_steps=20,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        model_name = f'dmvae_dataset{dataset_name}_seed{seed}_a{1e-5}_conflict'
        path = f'checkpoints/{model_name}.ckpt'
        trainer.save_checkpoint(path)

        dis_dmvae_model  = DisProbeFactory(dmvae_model)
        cml_dmvae_model  = ProbeFactory(dmvae_model, aggregation="cml")
        joint_dmvae_model = ProbeFactory(dmvae_model, aggregation="joint")
        
        dbf_fusion = LateFusionFactory(aggregation="dbf")
        cml_fusion = LateFusionFactory(aggregation="cml")
        avg_fusion = LateFusionFactory(aggregation="avg")
        
        models = [dis_dmvae_model, cml_dmvae_model, joint_dmvae_model, dbf_fusion, cml_fusion, avg_fusion]
        model_names = ['dmvae_dis','dmvae_cml','dmvae_joint','dbf_fusion','cml_fusion','avg_fusion']

        for model, name in zip(models,model_names):
            model_name = f'{name}_fusion_ds{dataset_name}_seed{seed}_conflict'
            print(f'Training model {model_name}')
            tags=[name,str(seed),dataset_name,'Conflict']
            model_parameters.update({"dataset": dataset_name, "seed": seed,'aggregation':name, 'UQ':'Conflict'})
            csv_logger = CSVLogger(
                save_dir="logs/",
                name=model_name,
            )
            gpus = 1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(
                accelerator="auto" if gpus else "cpu",
                devices=gpus if gpus else None,
                max_epochs=model_parameters['model_epochs'],
                log_every_n_steps=20, # For speed
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=csv_logger        # turn off TensorBoard for brevity
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            path = f'checkpoints/{model_name}.ckpt'
            trainer.save_checkpoint(path)
            print(f'Model {model_name}')
            pprint(test_metrics)
            if name == 'dmvae_dis':
                rows[seed]['Conflict'][dataset_name][name] = evaluate_subjective_model(model, test_loader)
            else:    
                rows[seed]['Conflict'][dataset_name][name] = evaluate_subjective_model_with_shared(model, test_loader)
            rows[seed]['Conflict'][dataset_name][name].update({'path':path})


df = build_metrics_dataframe_datasets(rows)
df['seed'] = df['seed'].astype(int)
df_main = df[['seed','type','dataset','model','view_0_evidence_mean','view_1_evidence_mean', 'shared_evidence_mean', 'fused_evidence_mean',
                  'view_0_aleatoric_mean', 'view_1_aleatoric_mean','shared_aleatoric_mean','fused_aleatoric_mean',
                  'view_0_epistemic_mean','view_1_epistemic_mean','shared_epistemic_mean','fused_epistemic_mean',
                 'view_0_accuracy',  'view_1_accuracy', 'shared_accuracy', 'fused_accuracy']]

df_grouped = df.groupby(['type','dataset','model']).mean().reset_index()
df_grouped.sort_values(by=['type','dataset','model'],inplace=True)
df_main_grouped = df_main.groupby(['type','dataset','model']).mean().reset_index()
df_main_grouped.sort_values(by=['type','dataset','model'],inplace=True)
with  pd.ExcelWriter('logs/dataset_analysis.xlsx') as writer:
    df_main_grouped.to_excel(writer, sheet_name='main_grouped',index=False)
    df.to_excel(writer, sheet_name='all_results',index=False)
    df_grouped.to_excel(writer,sheet_name='grouped_results',index=False)



