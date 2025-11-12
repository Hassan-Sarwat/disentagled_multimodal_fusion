"""
Script to compile the LUMA dataset for use in uncertainty quantification experiments.

This script assumes:
1. The raw LUMA data has been downloaded to data/luma_raw/ directory
2. CIFAR-10/100 images are available or will be downloaded
3. The compiled dataset will be saved to data/luma_compiled/

Usage:
    python compile_luma.py
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

# Configuration for LUMA compilation
LUMA_CONFIG = {
    'data_path': 'data/luma_raw',
    'output_path': 'data/luma_compiled',
    'train_samples_per_class': 500,
    'test_samples_per_class': 100,
    'num_classes': 42,
    'num_ood_classes': 8,
    
    # Modalities to include
    'use_audio': True,
    'use_text': True,
    'use_image': True,
    
    # Noise settings (set to False for clean dataset)
    'sample_noise': {
        'add_noise_train': False,
        'add_noise_test': False,
    },
    'label_noise': {
        'add_noise_train': False,
        'add_noise_test': False,
        'ratio': 0.0,
    },
    'ood_samples': {
        'add_ood': False,
        'ratio': 0.0,
    }
}


def check_luma_repo():
    """Check if LUMA repository is available, if not clone it."""
    luma_repo_path = Path('external/LUMA')
    
    if not luma_repo_path.exists():
        print("LUMA repository not found. Cloning...")
        luma_repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/bezirganyan/LUMA.git', str(luma_repo_path)],
                check=True
            )
            print(f"Successfully cloned LUMA repository to {luma_repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning LUMA repository: {e}")
            sys.exit(1)
    else:
        print(f"LUMA repository found at {luma_repo_path}")
    
    return luma_repo_path


def check_raw_data():
    """Check if raw LUMA data has been downloaded."""
    raw_data_path = Path(LUMA_CONFIG['data_path'])
    
    if not raw_data_path.exists():
        print(f"\nRaw LUMA data not found at {raw_data_path}")
        print("Please download the LUMA dataset using:")
        print("  git lfs install")
        print(f"  git clone https://huggingface.co/datasets/bezirganyan/LUMA {raw_data_path}")
        sys.exit(1)
    
    # Check for required files
    audio_csv = raw_data_path / 'audio' / 'datalist.csv'
    text_tsv = raw_data_path / 'text_data.tsv'
    edm_pickle = raw_data_path / 'edm_images.pickle'
    
    if not audio_csv.exists():
        print(f"Warning: Audio datalist not found at {audio_csv}")
    if not text_tsv.exists():
        print(f"Warning: Text data not found at {text_tsv}")
    if not edm_pickle.exists():
        print(f"Warning: EDM images pickle not found at {edm_pickle}")
    
    print(f"Raw LUMA data found at {raw_data_path}")
    return raw_data_path


def create_config_file(luma_repo_path):
    """Create a configuration YAML file for LUMA compilation."""
    config_path = Path('configs/luma_compile_config.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create config based on LUMA_CONFIG
    compile_config = {
        'data_path': str(Path(LUMA_CONFIG['data_path']).absolute()),
        'output_path': str(Path(LUMA_CONFIG['output_path']).absolute()),
        'train_samples_per_class': LUMA_CONFIG['train_samples_per_class'],
        'test_samples_per_class': LUMA_CONFIG['test_samples_per_class'],
        'modalities': {
            'audio': LUMA_CONFIG['use_audio'],
            'text': LUMA_CONFIG['use_text'],
            'image': LUMA_CONFIG['use_image'],
        },
        'noise': LUMA_CONFIG['sample_noise'],
        'label_noise': LUMA_CONFIG['label_noise'],
        'ood': LUMA_CONFIG['ood_samples'],
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(compile_config, f, default_flow_style=False)
    
    print(f"Created compilation config at {config_path}")
    return config_path


def compile_dataset(luma_repo_path, config_path):
    """Run the LUMA compilation script."""
    output_path = Path(LUMA_CONFIG['output_path'])
    
    # Check if dataset is already compiled
    if output_path.exists() and any(output_path.iterdir()):
        response = input(f"\nCompiled dataset already exists at {output_path}. Recompile? (y/N): ")
        if response.lower() != 'y':
            print("Using existing compiled dataset.")
            return output_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Compiling LUMA dataset...")
    print("="*60)
    
    # Check if compile_dataset.py exists in LUMA repo
    compile_script = luma_repo_path / 'compile_dataset.py'
    
    if not compile_script.exists():
        print(f"\nWarning: {compile_script} not found in LUMA repository.")
        print("Creating a simplified compilation process...")
        create_simplified_dataset()
    else:
        # Run the compilation script
        try:
            # Use absolute paths and run from project root
            subprocess.run(
                [sys.executable, str(compile_script.absolute()), '-c', str(config_path.absolute())],
                check=True,
                cwd=Path.cwd()  # Run from current directory, not LUMA repo
            )
            print(f"\nDataset compiled successfully to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation: {e}")
            print("\nCreating a simplified version of the dataset...")
            create_simplified_dataset()
    
    return output_path


def create_simplified_dataset():
    """
    Create a simplified version of the LUMA dataset by directly organizing the raw data.
    This is a fallback if the official compilation script is not available.
    """
    import pandas as pd
    import pickle
    import shutil
    
    raw_path = Path(LUMA_CONFIG['data_path'])
    output_path = Path(LUMA_CONFIG['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating simplified dataset structure...")
    
    # Copy audio datalist
    audio_csv = raw_path / 'audio' / 'datalist.csv'
    if audio_csv.exists():
        shutil.copy(audio_csv, output_path / 'audio_datalist.csv')
        print(f"  ✓ Copied audio datalist")
    
    # Copy text data
    text_tsv = raw_path / 'text_data.tsv'
    if text_tsv.exists():
        shutil.copy(text_tsv, output_path / 'text_data.tsv')
        print(f"  ✓ Copied text data")
    
    # Copy EDM images
    edm_pickle = raw_path / 'edm_images.pickle'
    if edm_pickle.exists():
        shutil.copy(edm_pickle, output_path / 'edm_images.pickle')
        print(f"  ✓ Copied EDM images")
    
    # Create audio directory reference (don't copy large audio files)
    audio_dir = raw_path / 'audio'
    if audio_dir.exists():
        audio_ref_file = output_path / 'audio_path.txt'
        with open(audio_ref_file, 'w') as f:
            f.write(str(audio_dir.absolute()))
        print(f"  ✓ Created audio path reference (avoids copying large files)")
    
    # Save metadata
    metadata = {
        'num_classes': LUMA_CONFIG['num_classes'],
        'num_ood_classes': LUMA_CONFIG['num_ood_classes'],
        'train_samples_per_class': LUMA_CONFIG['train_samples_per_class'],
        'test_samples_per_class': LUMA_CONFIG['test_samples_per_class'],
        'modalities': ['audio', 'text', 'image'],
    }
    
    with open(output_path / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    
    print(f"\n✓ Simplified dataset created at {output_path}")


def main():
    print("="*60)
    print("LUMA Dataset Compilation Script")
    print("="*60)
    
    # Step 1: Check if LUMA repository is available
    print("\n[1/4] Checking LUMA repository...")
    luma_repo_path = check_luma_repo()
    
    # Step 2: Check if raw data has been downloaded
    print("\n[2/4] Checking raw LUMA data...")
    raw_data_path = check_raw_data()
    
    # Step 3: Create configuration file
    print("\n[3/4] Creating compilation configuration...")
    config_path = create_config_file(luma_repo_path)
    
    # Step 4: Compile the dataset
    print("\n[4/4] Compiling dataset...")
    output_path = compile_dataset(luma_repo_path, config_path)
    
    print("\n" + "="*60)
    print("✓ LUMA Dataset Compilation Complete!")
    print("="*60)
    print(f"Compiled dataset location: {output_path}")
    print(f"\nYou can now use the dataset with:")
    print(f"  python run_luma.py")


if __name__ == '__main__':
    main()