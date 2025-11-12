"""
Main script for running experiments on the LUMA dataset.

This script trains and evaluates multiple fusion methods on the LUMA multimodal dataset:
1. DMVAE (Disentanglement-based Multimodal VAE)
2. DMVAE + Evidential Fusion (DCBF, DJOINT, PBF)
3. Late Fusion baselines (ABF, CBF, DBF)

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
import models.baselines as baselines
from models.dmvae import DMVAE
from models.evidential_probe import EvidentialProbeModule, DisentangledEvidentialProbeModule
from dataset_luma import get_luma_dataloaders
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe_datasets
from classifiers import IdentityEncoder

# Load configuration
CFG_PATH = Path("configs/luma_compile_config.yaml")
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


def train_dmvae(train_loader, seed, output_dims):
    """Train DMVAE for disentanglement."""
    dmvae_kwargs = {
        "feature_encoders": None,  # Will use default encoders
        "output_dim": output_dims,
        "dropout": C("dmvae.dropout", 0),
        "a": C("dmvae.a", 1e-5),
        "hidden_dim": C("dmvae.hidden_dim", 512),
        "embed_dim": C("dmvae.embed_dim", 200),
        "lr": C("dmvae.lr", 1e-4),
        "num_epochs": C("dmvae.num_epochs", 100),
    }
    
    dmvae = DMVAE(**dmvae_kwargs)
    
    trainer = pl.Trainer(
        accelerator=C("trainer.dmvae.accelerator", "auto"),
        devices=C("trainer.dmvae.devices", 1),
        max_epochs=dmvae_kwargs["num_epochs"],
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


def train_dmvae_fusion(dmvae, train_loader, test_loader, seed, aggregation="cml", fused=0):
    """Train evidential probe on DMVAE embeddings."""
    num_classes = C("dmvae_fusion.num_classes", 42)
    
    if aggregation in ["dcbf", "djoint", "pbf"]:
        # Disentangled fusion
        model = DisentangledEvidentialProbeModule(
            backbone=dmvae,
            num_classes=num_classes,
            input_dim=C("dmvae_fusion.input_dim", 200),
            aggregation=aggregation,
            dropout=C("dmvae_fusion.dropout", 0.1),
            annealing_start=C("dmvae_fusion.annealing_start", 10),
            lr=C("dmvae_fusion.lr", 3e-4),
            hidden_dim=tuple(C("dmvae_fusion.hidden_dim", [128])),
            freeze_backbone=True,
            fused=fused,
        )
    else:
        # Standard fusion
        model = EvidentialProbeModule(
            backbone=dmvae,
            num_classes=num_classes,
            input_dim=C("dmvae_fusion.input_dim", 200),
            aggregation=aggregation,
            dropout=C("dmvae_fusion.dropout", 0.1),
            annealing_start=C("dmvae_fusion.annealing_start", 10),
            lr=C("dmvae_fusion.lr", 3e-4),
            hidden_dim=tuple(C("dmvae_fusion.hidden_dim", [128])),
            freeze_backbone=True,
            fused=fused,
        )
    
    num_epochs = C("dmvae_fusion.num_epochs", 50)
    
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'dmvae_fusion_luma_seed{seed}_agg{aggregation}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    
    return model


def train_latefusion(train_loader, test_loader, seed, output_dims, aggregation="avg"):
    """Train late fusion baseline."""
    num_classes = C("latefusion.num_classes", 42)
    
    # Create classifiers for each modality
    classifiers = [
        (IdentityEncoder, {}),
        (IdentityEncoder, {}),
        (IdentityEncoder, {}),
    ]
    
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'latefusion_luma_seed{seed}_agg{aggregation}.ckpt'
    trainer.save_checkpoint(checkpoint_path)
    
    return model


def main():
    """Main execution function."""
    print("="*70)
    print("LUMA Dataset Experiments: Multimodal Uncertainty Quantification")
    print("="*70)
    
    # Get configuration
    seeds = C("experiment.seeds", [0, 1, 2, 3, 4])
    
    # Create output directory
    output_dir = Path(C("logging.output_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading LUMA dataset...")
    train_loader, test_loader, num_classes, num_views, dims = get_luma_data()
    print(f"  Classes: {num_classes}")
    print(f"  Modalities: {num_views}")
    print(f"  Dimensions: {dims}")
    
    # Store results
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running experiments with seed {seed} out of {len(seeds)}")
        print(f"{'='*70}")
        
        pl.seed_everything(seed)
        
        results[seed] = {}
        
        # Step 1: Train DMVAE for disentanglement
        print(f"\n[Seed {seed}] Training DMVAE...")
        dmvae = train_dmvae(train_loader, seed, output_dims=dims)
        
        # Step 2: Train DMVAE-based fusion methods
        print(f"\n[Seed {seed}] Training DMVAE Fusion methods...")
        
        # DCBF (Disentangled Cumulative Belief Fusion)
        print(f"  - DCBF (Disentangled Cumulative Belief Fusion)")
        dcbf_model = train_dmvae_fusion(dmvae, train_loader, test_loader, seed, aggregation="dcbf")
        results[seed]['dcbf'] = evaluate_subjective_model_with_shared(dcbf_model, test_loader)
        
        # DJOINT (Disentangled Joint)
        print(f"  - DJOINT (Disentangled Joint)")
        djoint_model = train_dmvae_fusion(dmvae, train_loader, test_loader, seed, aggregation="djoint")
        results[seed]['djoint'] = evaluate_subjective_model_with_shared(djoint_model, test_loader)
        
        # PBF (Private Belief Fusion)
        print(f"  - PBF (Private Only)")
        pbf_model = train_dmvae_fusion(dmvae, train_loader, test_loader, seed, aggregation="pbf")
        results[seed]['pbf'] = evaluate_subjective_model_with_shared(pbf_model, test_loader)
        
        # Step 3: Train Late Fusion baselines
        print(f"\n[Seed {seed}] Training Late Fusion baselines...")
        
        # Average Belief Fusion
        print(f"  - ABF (Average Belief Fusion)")
        abf_model = train_latefusion(train_loader, test_loader, seed, dims, aggregation="avg")
        results[seed]['abf'] = evaluate_subjective_model(abf_model, test_loader)
        
        # Cumulative Belief Fusion
        print(f"  - CBF (Cumulative Belief Fusion)")
        cbf_model = train_latefusion(train_loader, test_loader, seed, dims, aggregation="cml")
        results[seed]['cbf'] = evaluate_subjective_model(cbf_model, test_loader)
        
        # Discounted Belief Fusion
        print(f"  - DBF (Discounted Belief Fusion)")
        dbf_model = train_latefusion(train_loader, test_loader, seed, dims, aggregation="dsc")
        results[seed]['dbf'] = evaluate_subjective_model(dbf_model, test_loader)
        
        print(f"\n[Seed {seed}] Completed!")
    
    # Build results dataframe
    print("\n" + "="*70)
    print("Building results dataframe...")
    print("="*70)
    
    # Organize results for DataFrame
    nested_results = {}
    for seed in seeds:
        nested_results[seed] = {
            'luma': {
                'LUMA': results[seed]
            }
        }
    
    df = build_metrics_dataframe_datasets(nested_results)
    
    # Compute grouped statistics
    df_grouped = df.groupby(['dataset', 'model']).agg(['mean', 'std']).reset_index()
    
    # Save results
    excel_path = output_dir / C("logging.excel_filename", "luma_results.xlsx")
    print(f"\nSaving results to {excel_path}")
    
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='all_results', index=False)
        df_grouped.to_excel(writer, sheet_name='grouped_results', index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Display key metrics
    metrics_to_show = ['fused_accuracy', 'fused_evidence_mean', 'fused_aleatoric_mean', 'fused_epistemic_mean']
    
    for metric in metrics_to_show:
        if metric in df.columns:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            summary = df.groupby('model')[metric].agg(['mean', 'std'])
            print(summary.to_string())
    
    print(f"\n{'='*70}")
    print(f"✓ Experiments completed successfully!")
    print(f"  Results saved to: {excel_path}")
    print(f"  Checkpoints saved to: {Path(C('logging.checkpoint_dir', 'checkpoints'))}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()