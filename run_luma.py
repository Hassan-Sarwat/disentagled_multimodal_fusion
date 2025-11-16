import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from pprint import pprint
from functools import partial

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Import models
import models.baselines as baselines
from models.dmvae import DMVAE
from models.evidential_probe import EvidentialProbeModule, DisentangledEvidentialProbeModule
from models.classifiers import IdentityEncoder, ImageEncoder, AudioEncoder, TextEncoder

from datasets.dataset_luma import get_luma_dataloaders
from analysis import (
    evaluate_subjective_model, 
    evaluate_subjective_model_with_shared, 
    build_metrics_dataframe_datasets
)


# ============================================================================
# Configuration Loading
# ============================================================================

CFG_PATH = Path("configs/luma_config.yaml")
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


# ============================================================================
# Data Loading
# ============================================================================

def get_luma_data():
    """Load LUMA dataset with train/test loaders."""
    data_path = C("data.luma_path", "data/luma_compiled")
    batch_size = C("dataloader.batch_size", 64)
    num_workers = C("dataloader.num_workers", 4)
    
    # Audio configuration
    audio_config = {
        'sample_rate': C("data.audio.sample_rate", 16000),
        'max_length': C("data.audio.max_length", 3.0),
        'n_mfcc': C("data.audio.n_mfcc", 40),
        'use_mfcc': C("data.audio.use_mfcc", True),
    }
    
    # Text configuration
    text_config = {
        'max_length': C("data.text.max_length", 128),
        'model_name': C("data.text.model_name", "bert-base-uncased"),
        'use_pretrained': C("data.text.use_pretrained", True),
    }
    
    # Image configuration
    image_config = {
        'size': tuple(C("data.image.size", [32, 32])),
        'normalize': C("data.image.normalize", True),
    }
    
    train_loader, test_loader, num_classes, num_views, dims = get_luma_dataloaders(
        data_path=data_path,
        audio_config=audio_config,
        text_config=text_config,
        image_config=image_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader, num_classes, num_views, dims


# ============================================================================
# Model Factory Functions
# ============================================================================

def build_factories(model_params, probe_input_dim, dmvae_kwargs):
    """
    Build factory functions for all model types using functools.partial.
    This matches the pattern used in run.py for consistency.
    """
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


# ============================================================================
# Main Experiment Function
# ============================================================================

def main():
    """Main experiment function - runs all experiments."""
    
    # Experiment Configuration
    seeds = C("experiment.seeds", [0, 1, 2, 3, 4])
    
    # LUMA-specific learning rate (can be configured in YAML)
    luma_lr = C("optim.luma_lr", 3e-4)
    
    # Model parameters
    model_parameters = {
        "dropout_p": C("probes.dropout_p", 0.1),
        "annealing_start": C("probes.annealing_start", 50),
        "model_epochs": 2,# C("probes.model_epochs", 200),
        "model_hidden_dim": tuple(C("probes.model_hidden_dim", (128,))),
    }
    
    probe_input_dim = C("probes.input_dim", 200)
    
    # DMVAE parameters
    dmvae_kwargs = {
        "dropout": C("dmvae.dropout", 0),
        "a": C("dmvae.a", 1e-5),
        "hidden_dim": C("dmvae.hidden_dim", 512),
        "embed_dim": C("dmvae.embed_dim", 200),
        "lr": C("dmvae.lr", 1e-4),
        "num_epochs": 3 #C("dmvae.num_epochs", 100),
    }
    
    rows = {}
    
    for seed in seeds:
        pl.seed_everything(seed)
        rows[seed] = {}
        
        # ====================================================================
        # Load LUMA Dataset
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"[Seed {seed}] Loading LUMA dataset...")
        print(f"{'='*70}")
        
        rows[seed]['Normal'] = {}
        rows[seed]['Normal']['LUMA'] = {}
        
        train_loader, test_loader, num_classes, num_views, dims = get_luma_data()
        
        # Update model parameters with dataset-specific info
        model_parameters["classes"] = num_classes
        model_parameters["output_dims"] = dims
        model_parameters['classifiers'] = [
            (AudioEncoder, {'input_dim': 40, 'output_dim': 200, 'dropout': 0.1}),
            (TextEncoder, {'input_dim': 128, 'output_dim': 200, 'dropout': 0.1}),
            (ImageEncoder, {'output_dim': 200, 'dropout': 0.1})
        ]
        model_parameters['lr'] = luma_lr
        
        print(f"  Dataset: LUMA")
        print(f"  Classes: {num_classes}")
        print(f"  Views: {num_views}")
        print(f"  Dimensions: {dims}")
        
        # ====================================================================
        # Build Model Factories
        # ====================================================================
        DMVAEFactory, ProbeFactory, DisProbeFactory, LateFusionFactory = build_factories(
            model_parameters, probe_input_dim, dmvae_kwargs
        )
        
        # ====================================================================
        # Train DMVAE (Disentanglement Backbone)
        # ====================================================================
        print(f"\n[Seed {seed}] Training DMVAE...")
        
        dmvae_model = DMVAEFactory()
        
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=dmvae_kwargs["num_epochs"],
            enable_progress_bar=True,
            log_every_n_steps=20,
            enable_model_summary=False,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        
        # Save DMVAE checkpoint
        dmvae_checkpoint = f'checkpoints/dmvae_datasetLUMA_seed{seed}_a{dmvae_kwargs["a"]}_normal.ckpt'
        os.makedirs('checkpoints', exist_ok=True)
        trainer.save_checkpoint(dmvae_checkpoint)
        print(f"  ✓ Saved DMVAE: {dmvae_checkpoint}")
        
        # ====================================================================
        # Create All Fusion Models
        # ====================================================================
        print(f"\n[Seed {seed}] Creating fusion models...")
        
        # Disentangled models (use DMVAE embeddings)
        dis_dmvae_model = DisProbeFactory(dmvae_model)
        cml_dmvae_model = ProbeFactory(dmvae_model, aggregation="cml")
        joint_dmvae_model = ProbeFactory(dmvae_model, aggregation="joint")
        
        # Late fusion baselines (no disentanglement)
        dbf_fusion = LateFusionFactory(aggregation="dbf")
        cml_fusion = LateFusionFactory(aggregation="cml")
        avg_fusion = LateFusionFactory(aggregation="avg")
        
        models = [
            dis_dmvae_model, 
            cml_dmvae_model, 
            joint_dmvae_model, 
            dbf_fusion, 
            cml_fusion, 
            avg_fusion
        ]
        model_names = [
            'dmvae_dis', 
            'dmvae_cml', 
            'dmvae_joint', 
            'dbf_fusion', 
            'cml_fusion', 
            'avg_fusion'
        ]
        
        # ====================================================================
        # Train and Evaluate All Models
        # ====================================================================
        for model, name in zip(models, model_names):
            model_name = f'{name}_fusion_dsLUMA_seed{seed}'
            print(f'\n[Seed {seed}] Training {model_name}...')
            
            csv_logger = CSVLogger(
                save_dir="logs/",
                name=model_name,
            )
            
            gpus = 1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(
                accelerator="auto" if gpus else "cpu",
                devices=gpus if gpus else None,
                max_epochs=model_parameters['model_epochs'],
                log_every_n_steps=20,
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=csv_logger,
            )
            
            # Train
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            
            # Save checkpoint
            path = f'checkpoints/{model_name}.ckpt'
            trainer.save_checkpoint(path)
            
            # Test
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            print(f'  Test results for {model_name}:')
            pprint(test_metrics)
            
            # Evaluate and store results
            if name == 'dmvae_dis':
                # Private only - no shared modality
                rows[seed]['Normal']['LUMA'][name] = evaluate_subjective_model(model, test_loader)
            else:
                # Includes shared modality
                rows[seed]['Normal']['LUMA'][name] = evaluate_subjective_model_with_shared(model, test_loader)
            
            rows[seed]['Normal']['LUMA'][name].update({'path': path})
            print(f'  ✓ Completed {model_name}')
    
    
    # ========================================================================
    # Build and Save Results DataFrame
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Building results dataframe...")
    print(f"{'='*70}")
    
    df = build_metrics_dataframe_datasets(rows)
    df['seed'] = df['seed'].astype(int)
    
    # Select main columns (matching run.py structure)
    df_main = df[[
        'seed', 'type', 'dataset', 'model',
        'view_0_evidence_mean', 'view_1_evidence_mean', 'shared_evidence_mean', 'fused_evidence_mean',
        'view_0_aleatoric_mean', 'view_1_aleatoric_mean', 'shared_aleatoric_mean', 'fused_aleatoric_mean',
        'view_0_epistemic_mean', 'view_1_epistemic_mean', 'shared_epistemic_mean', 'fused_epistemic_mean',
        'view_0_accuracy', 'view_1_accuracy', 'shared_accuracy', 'fused_accuracy'
    ]]
    
    # Compute grouped statistics (mean across seeds)
    df_grouped = df.groupby(['type', 'dataset', 'model']).mean().reset_index()
    df_grouped.sort_values(by=['type', 'dataset', 'model'], inplace=True)
    
    df_main_grouped = df_main.groupby(['type', 'dataset', 'model']).mean().reset_index()
    df_main_grouped.sort_values(by=['type', 'dataset', 'model'], inplace=True)
    
    # Save to Excel
    os.makedirs('logs', exist_ok=True)
    excel_path = 'logs/luma_analysis.xlsx'
    
    with pd.ExcelWriter(excel_path) as writer:
        df_main_grouped.to_excel(writer, sheet_name='main_grouped', index=False)
        df.to_excel(writer, sheet_name='all_results', index=False)
        df_grouped.to_excel(writer, sheet_name='grouped_results', index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Experiments completed successfully!")
    print(f"  Results saved to: {excel_path}")
    print(f"{'='*70}\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    main()