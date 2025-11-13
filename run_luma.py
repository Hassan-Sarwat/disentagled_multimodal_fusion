"""
Main script for running experiments on the LUMA dataset.

This script trains and evaluates multiple fusion methods on the LUMA multimodal dataset:
1. DMVAE (Disentanglement-based Multimodal VAE)
2. Disentangled fusion methods:
   - DCBF (Disentangled Cumulative Belief Fusion)
   - DJOINT (Disentangled Joint)
   - PBF (Private Belief Fusion - ablation)
3. Late Fusion baselines (without disentanglement):
   - avg (Average Belief Fusion)
   - CBF (Cumulative Belief Fusion)
   - DBF (Discounted Belief Fusion)

Usage:
    python run_luma.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from collections import OrderedDict

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Import models
import models.baselines as baselines
from models.dmvae import DMVAE
from models.evidential_probe import EvidentialProbeModule, DisentangledEvidentialProbeModule

# Import dataset
from dataset_luma import get_luma_dataloaders

# Import analysis utilities
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe_datasets

# Import classifiers
from classifiers import IdentityEncoder

# Load configuration
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


def get_luma_data():
    """Load LUMA dataset."""
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


def train_dmvae(train_loader, seed, num_classes, output_dims):
    """Train DMVAE for disentanglement."""
    print(f"  Training DMVAE...")
    
    # Create feature encoders (IdentityEncoder for each modality)
    feature_encoders = [(IdentityEncoder, {}) for _ in range(len(output_dims))]
    
    dmvae = DMVAE(
        feature_encoders=feature_encoders,
        output_dim=output_dims,
        dropout=C("dmvae.dropout", 0),
        a=C("dmvae.a", 1e-5),
        hidden_dim=C("dmvae.hidden_dim", 512),
        embed_dim=C("dmvae.embed_dim", 200),
        lr=C("dmvae.lr", 1e-4),
        num_epochs=C("dmvae.num_epochs", 100),
    )
    
    trainer = pl.Trainer(
        accelerator=C("trainer.dmvae.accelerator", "auto"),
        devices=C("trainer.dmvae.devices", 1),
        max_epochs=C("dmvae.num_epochs", 100),
        log_every_n_steps=C("trainer.dmvae.log_every_n_steps", 20),
        enable_progress_bar=C("trainer.dmvae.enable_progress_bar", True),
        enable_model_summary=False,
    )
    
    trainer.fit(dmvae, train_dataloaders=train_loader)
    
    # Save checkpoint
    checkpoint_dir = Path(C("logging.checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'dmvae_luma_seed{seed}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    
    return dmvae


def train_disentangled_fusion(dmvae, train_loader, test_loader, seed, num_classes, aggregation="dcbf"):
    """
    Train disentangled evidential fusion methods.
    
    Uses:
    - DisentangledEvidentialProbeModule for PBF (private only)
    - EvidentialProbeModule for DCBF and DJOINT
    """
    print(f"  Training {aggregation.upper()}...")
    
    # Map aggregation names
    agg_map = {
        'dcbf': 'cml',    # Disentangled Cumulative Belief Fusion
        'djoint': 'cml', # Disentangled Joint
        'pbf': 'cml',     # Private Belief Fusion (ablation)
    }
    
    agg_name = agg_map.get(aggregation, aggregation)
    
    # Use DisentangledEvidentialProbeModule for PBF (private only)
    if aggregation == 'pbf':
        model = DisentangledEvidentialProbeModule(
            backbone=dmvae,
            num_classes=num_classes,
            input_dim=C("probes.input_dim", 200),
            dropout=C("probes.dropout_p", 0.1),
            annealing_start=C("probes.annealing_start", 50),
            lr=C("probes.lr", 3e-4),
            hidden_dim=tuple(C("probes.model_hidden_dim", [128])),
            freeze_backbone=True,
        )
    else:
        # Use EvidentialProbeModule for DCBF and DJOINT
        model = EvidentialProbeModule(
            backbone=dmvae,
            num_classes=num_classes,
            input_dim=C("probes.input_dim", 200),
            aggregation=agg_name,
            dropout=C("probes.dropout_p", 0.1),
            annealing_start=C("probes.annealing_start", 50),
            lr=C("probes.lr", 3e-4),
            hidden_dim=tuple(C("probes.model_hidden_dim", [128])),
            freeze_backbone=True,
            fused=1,  # Include shared modality
        )
    
    num_epochs = C("probes.model_epochs", 200)
    
    trainer = pl.Trainer(
        accelerator=C("trainer.fusion.accelerator", "auto"),
        devices=C("trainer.fusion.devices", 1),
        max_epochs=num_epochs,
        log_every_n_steps=C("trainer.fusion.log_every_n_steps", 20),
        enable_progress_bar=C("trainer.fusion.enable_progress_bar", True),
        enable_model_summary=C("trainer.fusion.enable_model_summary", False),
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    # Save checkpoint
    checkpoint_dir = Path(C("logging.checkpoint_dir", "checkpoints"))
    checkpoint_path = checkpoint_dir / f'{aggregation}_luma_seed{seed}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    
    return model


def train_late_fusion(train_loader, test_loader, seed, num_classes, output_dims, aggregation="avg"):
    """
    Train late fusion baselines (without disentanglement).
    
    Uses LateFusion from baselines module.
    """
    print(f"  Training {aggregation.upper()} (late fusion)...")
    
    
    # Create classifiers for each modality (IdentityEncoder)
    classifiers = [(IdentityEncoder, {}) for _ in range(len(output_dims))]
    
    model = baselines.LateFusion(
        classifiers,
        output_dims,
        num_classes=num_classes,
        dropout=C("latefusion.dropout", 0.1),
        aggregation=aggregation,
        annealing_start=C("latefusion.annealing_start", 10),
        lr=C("latefusion.lr", 3e-4),
        hidden_dim=tuple(C("latefusion.hidden_dim", [128])),
        fused=0,
    )
    
    num_epochs = C("latefusion.num_epochs", 50)
    
    trainer = pl.Trainer(
        accelerator=C("trainer.fusion.accelerator", "auto"),
        devices=C("trainer.fusion.devices", 1),
        max_epochs=num_epochs,
        log_every_n_steps=C("trainer.fusion.log_every_n_steps", 20),
        enable_progress_bar=C("trainer.fusion.enable_progress_bar", True),
        enable_model_summary=C("trainer.fusion.enable_model_summary", False),
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    # Save checkpoint
    checkpoint_dir = Path(C("logging.checkpoint_dir", "checkpoints"))
    checkpoint_path = checkpoint_dir / f'{aggregation}_luma_seed{seed}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    
    return model


def main():
    """Main execution function."""
    print("="*70)
    print("LUMA Dataset Experiments: Multimodal Uncertainty Quantification")
    print("="*70)
    
    # Get configuration
    seeds = C("experiment.seeds", [0])
    
    # Create output directory
    output_dir = Path(C("logging.output_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(C("logging.checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading LUMA dataset...")
    train_loader, test_loader, num_classes, num_views, dims = get_luma_data()
    print(f"  Classes: {num_classes}")
    print(f"  Modalities: {num_views}")
    print(f"  Dimensions: {dims}")
    
    # Model parameters
    model_parameters = {
        "classes": num_classes,
        "output_dims": dims,
        "classifiers": [(IdentityEncoder, {}) for _ in range(len(dims))],
        "lr": C("probes.lr", 3e-4),
        "dropout_p": C("probes.dropout_p", 0.1),
        "annealing_start": C("probes.annealing_start", 50),
        "model_epochs": C("probes.model_epochs", 200),
        "model_hidden_dim": tuple(C("probes.model_hidden_dim", [128])),
    }
    
    probe_input_dim = C("probes.input_dim", 200)
    
    dmvae_kwargs = {
        "dropout": C("dmvae.dropout", 0),
        "a": C("dmvae.a", 1e-5),
        "hidden_dim": C("dmvae.hidden_dim", 512),
        "embed_dim": C("dmvae.embed_dim", 200),
        "lr": C("dmvae.lr", 1e-4),
        "num_epochs": C("dmvae.num_epochs", 2),
    }
    
    # Store results
    rows = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running experiments with seed {seed}")
        print(f"{'='*70}")
        
        pl.seed_everything(seed)
        
        rows[seed] = {}
        rows[seed]['Normal'] = {}
        rows[seed]['Normal']['LUMA'] = {}
        
        # ========================================
        # Step 1: Train DMVAE for disentanglement
        # ========================================
        print(f"\n[Seed {seed}] Training DMVAE...")
        
        dmvae_model = DMVAE(
            feature_encoders=model_parameters["classifiers"],
            output_dim=model_parameters["output_dims"],
            dropout=dmvae_kwargs["dropout"],
            a=dmvae_kwargs["a"],
            hidden_dim=dmvae_kwargs["hidden_dim"],
            embed_dim=dmvae_kwargs["embed_dim"],
            lr=dmvae_kwargs["lr"],
            num_epochs=dmvae_kwargs["num_epochs"],
        )
        
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=dmvae_kwargs["num_epochs"],
            enable_progress_bar=True,
            log_every_n_steps=20,
            enable_model_summary=False,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        
        dmvae_checkpoint = checkpoint_dir / f'dmvae_datasetLUMA_seed{seed}_a{dmvae_kwargs["a"]}_normal.ckpt'
        trainer.save_checkpoint(dmvae_checkpoint)
        print(f"  Saved DMVAE checkpoint: {dmvae_checkpoint}")
        
        # ========================================
        # Step 2: Create all models
        # ========================================
        print(f"\n[Seed {seed}] Creating fusion models...")
        
        # Disentangled models (use DMVAE embeddings)
        dis_dmvae_model = DisentangledEvidentialProbeModule(
            backbone=dmvae_model,
            num_classes=model_parameters["classes"],
            input_dim=probe_input_dim,
            lr=model_parameters["lr"],
            annealing_start=model_parameters["annealing_start"],
            hidden_dim=model_parameters["model_hidden_dim"],
            dropout=model_parameters["dropout_p"],
            freeze_backbone=True,
        )
        
        cml_dmvae_model = EvidentialProbeModule(
            backbone=dmvae_model,
            num_classes=model_parameters["classes"],
            input_dim=probe_input_dim,
            aggregation="cml",
            lr=model_parameters["lr"],
            annealing_start=model_parameters["annealing_start"],
            hidden_dim=model_parameters["model_hidden_dim"],
            dropout=model_parameters["dropout_p"],
            freeze_backbone=True,
            fused=0,
        )
        
        joint_dmvae_model = EvidentialProbeModule(
            backbone=dmvae_model,
            num_classes=model_parameters["classes"],
            input_dim=probe_input_dim,
            aggregation="joint",
            lr=model_parameters["lr"],
            annealing_start=model_parameters["annealing_start"],
            hidden_dim=model_parameters["model_hidden_dim"],
            dropout=model_parameters["dropout_p"],
            freeze_backbone=True,
            fused=0,
        )
        
        # Late fusion models (no disentanglement)
        dbf_fusion = baselines.LateFusion(
            model_parameters["classifiers"],
            model_parameters["output_dims"],
            model_parameters["classes"],
            dropout=model_parameters["dropout_p"],
            aggregation="dbf",
            annealing_start=10,
            lr=model_parameters["lr"],
            hidden_dim=model_parameters["model_hidden_dim"],
            fused=0,
        )
        
        cml_fusion = baselines.LateFusion(
            model_parameters["classifiers"],
            model_parameters["output_dims"],
            model_parameters["classes"],
            dropout=model_parameters["dropout_p"],
            aggregation="cml",
            annealing_start=10,
            lr=model_parameters["lr"],
            hidden_dim=model_parameters["model_hidden_dim"],
            fused=0,
        )
        
        avg_fusion = baselines.LateFusion(
            model_parameters["classifiers"],
            model_parameters["output_dims"],
            model_parameters["classes"],
            dropout=model_parameters["dropout_p"],
            aggregation="avg",
            annealing_start=10,
            lr=model_parameters["lr"],
            hidden_dim=model_parameters["model_hidden_dim"],
            fused=0,
        )
        
        models = [dis_dmvae_model, cml_dmvae_model, joint_dmvae_model, dbf_fusion, cml_fusion, avg_fusion]
        model_names = ['dmvae_dis', 'dmvae_cml', 'dmvae_joint', 'dbf_fusion', 'cml_fusion', 'avg_fusion']
        
        # ========================================
        # Step 3: Train and evaluate all models
        # ========================================
        for model, name in zip(models, model_names):
            model_name = f'{name}_fusion_dsLUMA_seed{seed}'
            print(f'\nTraining model {model_name}')
            
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
            
            # Train model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'{model_name}.ckpt'
            trainer.save_checkpoint(checkpoint_path)
            
            # Test model
            print(f'Evaluating model {model_name}...')
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            print(f'Test metrics for {model_name}:')
            from pprint import pprint
            pprint(test_metrics)
            
            # Evaluate and store results
            if name == 'dmvae_dis':
                # Private only - no shared modality
                rows[seed]['Normal']['LUMA'][name] = evaluate_subjective_model(model, test_loader)
            else:
                # Includes shared modality
                rows[seed]['Normal']['LUMA'][name] = evaluate_subjective_model_with_shared(model, test_loader)
            
            rows[seed]['Normal']['LUMA'][name].update({'path': str(checkpoint_path)})
            print(f'✓ Completed {model_name}\n')
        
        print(f"\n[Seed {seed}] ✓ Completed all experiments!")
    
    # ========================================
    # Build and Save Results
    # ========================================
    print("\n" + "="*70)
    print("Building results dataframe...")
    print("="*70)
    
    # Build dataframe using the same structure as run.py
    df = build_metrics_dataframe_datasets(rows)
    
    # Compute grouped statistics (mean and std across seeds)
    print("\nComputing statistics across seeds...")
    
    grouped_cols = ['type', 'dataset', 'model']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['seed']]
    
    df_grouped = df.groupby(grouped_cols)[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Save results
    excel_path = output_dir / C("logging.excel_filename", "luma_results.xlsx")
    print(f"\nSaving results to {excel_path}")
    
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='all_results', index=False)
        df_grouped.to_excel(writer, sheet_name='grouped_results', index=False)
    
    # ========================================
    # Print Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Display key metrics
    key_metrics = [
        'fused_accuracy',
        'fused_evidence_mean',
        'fused_aleatoric_mean',
        'fused_epistemic_mean'
    ]
    
    print("\nResults by model (mean ± std across seeds):")
    print("-" * 70)
    
    for metric in key_metrics:
        if metric in df.columns:
            print(f"\n{metric.replace('_', ' ').upper()}:")
            summary = df.groupby('model')[metric].agg(['mean', 'std'])
            for model_name in summary.index:
                mean_val = summary.loc[model_name, 'mean']
                std_val = summary.loc[model_name, 'std']
                print(f"  {model_name:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Experiments completed successfully!")
    print(f"  Results saved to: {excel_path}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()