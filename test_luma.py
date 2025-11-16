"""
Test script to verify LUMA dataset compilation and loading.

Run this after compile_luma.py to ensure everything works correctly.

Usage:
    python test_luma.py
"""

import sys
from pathlib import Path
import torch

def test_compilation():
    """Test that compilation produced the required files."""
    print("="*60)
    print("TEST 1: Checking compiled dataset files")
    print("="*60)
    
    required_files = [
        'data/luma_compiled/audio_datalist.csv',
        'data/luma_compiled/text_data.tsv',
        'data/luma_compiled/edm_images.pickle',
        'data/luma_compiled/metadata.yaml',
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    # Check for audio reference
    audio_ref = Path('data/luma_compiled/audio_path.txt')
    if audio_ref.exists():
        print(f"  ✓ audio_path.txt (references: {audio_ref.read_text().strip()})")
    else:
        print(f"  ✗ audio_path.txt")
        all_exist = False
    
    if all_exist:
        print("\n✓ All required files present!")
        return True
    else:
        print("\n✗ Some files are missing. Run compile_luma.py first.")
        return False


def test_dataset_loading():
    """Test loading the dataset."""
    print("\n" + "="*60)
    print("TEST 2: Loading dataset")
    print("="*60)
    
    try:
        from datasets.dataset_luma import get_luma_dataloaders
        
        print("  Creating dataloaders...")
        train_loader, test_loader, num_classes, num_views, dims = get_luma_dataloaders(
            data_path="data/luma_compiled",
            batch_size=4,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
        )
        
        print(f"  ✓ Dataset loaded successfully")
        print(f"    - Classes: {num_classes}")
        print(f"    - Modalities: {num_views}")
        print(f"    - Dimensions: {dims}")
        print(f"    - Train batches: {len(train_loader)}")
        print(f"    - Test batches: {len(test_loader)}")
        
        return True, (train_loader, test_loader)
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batch_loading(loaders):
    """Test loading a single batch."""
    print("\n" + "="*60)
    print("TEST 3: Loading sample batch")
    print("="*60)
    
    if loaders is None:
        print("  ✗ Skipping (dataset not loaded)")
        return False
    
    train_loader, test_loader = loaders
    
    try:
        print("  Loading first training batch...")
        batch = next(iter(train_loader))
        views, labels = batch
        
        print(f"  ✓ Batch loaded successfully")
        print(f"    - Audio shape: {views[0].shape}")
        print(f"    - Text shape: {views[1].shape}")
        print(f"    - Image shape: {views[2].shape}")
        print(f"    - Labels shape: {labels.shape}")
        print(f"    - Labels range: [{labels.min()}, {labels.max()}]")
        
        # Verify shapes make sense
        batch_size = labels.shape[0]
        assert views[0].shape[0] == batch_size, "Audio batch size mismatch"
        assert views[1].shape[0] == batch_size, "Text batch size mismatch"
        assert views[2].shape[0] == batch_size, "Image batch size mismatch"
        
        print(f"  ✓ Batch shapes are consistent")
        
        # Check data types
        print(f"    - Audio dtype: {views[0].dtype}")
        print(f"    - Text dtype: {views[1].dtype}")
        print(f"    - Image dtype: {views[2].dtype}")
        print(f"    - Labels dtype: {labels.dtype}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_loading():
    """Test audio loading specifically."""
    print("\n" + "="*60)
    print("TEST 4: Testing audio loading")
    print("="*60)
    
    try:
        import pandas as pd
        import torchaudio
        
        # Check audio reference
        audio_ref_path = Path('data/luma_compiled/audio_path.txt')
        if not audio_ref_path.exists():
            print("  ✗ audio_path.txt not found")
            return False
        
        audio_dir = Path(audio_ref_path.read_text().strip())
        print(f"  Audio directory: {audio_dir}")
        
        if not audio_dir.exists():
            print(f"  ✗ Audio directory does not exist: {audio_dir}")
            return False
        
        print(f"  ✓ Audio directory exists")
        
        # Load audio datalist
        datalist = pd.read_csv('data/luma_compiled/audio_datalist.csv')
        print(f"  ✓ Audio datalist loaded: {len(datalist)} entries")
        
        # Try loading first audio file
        first_audio = datalist.iloc[0]
        # The path in datalist is like 'cv_audio/bird/103.wav'
        # The audio_dir is like 'data/luma_raw/audio'
        # So full path is: audio_dir / path
        audio_path = audio_dir / first_audio['path']
        
        print(f"  Testing first audio file: {audio_path.name}")
        
        if not audio_path.exists():
            print(f"  ✗ Audio file not found: {audio_path}")
            print(f"     Datalist path: {first_audio['path']}")
            print(f"     Looking for: {audio_path}")
            
            # Try to find ANY audio file to verify structure
            print(f"\n  Checking audio directory structure...")
            if audio_dir.exists():
                subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
                print(f"  Subdirectories in {audio_dir.name}:")
                for subdir in subdirs[:5]:  # Show first 5
                    wav_files = list(subdir.glob('**/*.wav'))
                    print(f"    {subdir.name}/ ({len(wav_files)} .wav files)")
                    if wav_files:
                        print(f"      Example: {wav_files[0].relative_to(audio_dir)}")
            return False
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        print(f"  ✓ Audio loaded successfully")
        print(f"    - Shape: {waveform.shape}")
        print(f"    - Sample rate: {sample_rate}")
        print(f"    - Duration: {waveform.shape[1] / sample_rate:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LUMA Dataset Test Suite")
    print("="*70)
    
    results = []
    
    # Test 1: Compilation
    results.append(("Compilation", test_compilation()))
    
    if not results[-1][1]:
        print("\n" + "="*70)
        print("❌ Tests stopped - compilation incomplete")
        print("="*70)
        print("\nPlease run: python compile_luma.py")
        return False
    
    # Test 2: Dataset loading
    success, loaders = test_dataset_loading()
    results.append(("Dataset Loading", success))
    
    if success:
        # Test 3: Batch loading
        results.append(("Batch Loading", test_batch_loading(loaders)))
        
        # Test 4: Audio loading
        results.append(("Audio Loading", test_audio_loading()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed! You can now run: python run_luma.py")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)