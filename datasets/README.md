# LUMA Dataset Setup

This section explains how to download and compile the LUMA multimodal dataset for use in this project.

## Prerequisites

Before compiling the LUMA dataset, ensure you have:

- **Git LFS** (Large File Storage) installed
  ```bash
  # Install git-lfs if not already installed
  # On Ubuntu/Debian:
  sudo apt-get install git-lfs
  
  # On macOS:
  brew install git-lfs
  
  # On Windows (using Chocolatey):
  choco install git-lfs
  
  # Initialize git-lfs
  git lfs install
  ```

- **Python dependencies** installed
  ```bash
  pip install -r requirements.txt
  ```

- **NLTK data** (will be auto-downloaded if missing)

---

## Quick Start

If you just forked this repository and want to set up the LUMA dataset:

```bash
# 1. Download the raw LUMA data from HuggingFace
git lfs install
git clone https://huggingface.co/datasets/bezirganyan/LUMA data/luma_raw

# 2. Compile the dataset
python compile_luma.py

# 3. Verify the compilation
python test_luma.py

# 4. Run experiments
python run_luma.py
```

That's it! The compilation script handles everything automatically.

---

## Detailed Setup Instructions

### Step 1: Download Raw LUMA Data

The LUMA dataset is hosted on HuggingFace. Download it using Git LFS:

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Clone the LUMA dataset from HuggingFace
git clone https://huggingface.co/datasets/bezirganyan/LUMA data/luma_raw
```

**What gets downloaded:**
- `audio/` - Audio files (.wav) with pronunciations of class labels
  - From Mozilla Common Voice, Spoken Wikipedia, and LibriSpeech
  - `datalist.csv` - Index of all audio files
- `text_data.tsv` - Text passages generated using Gemma-7B LLM
- `edm_images.pickle` - Generated images from EDM (Energy-based Diffusion Models)
  - Based on CIFAR-10/100 distribution

**Expected download size:** ~5-10 GB (depending on modalities)

**Note:** The raw data is in `.gitignore` and won't be committed to your repository.

---

### Step 2: Compile the Dataset

The compilation process organizes the raw data into a format optimized for training:

```bash
python compile_luma.py
```

**What this script does:**
1. ✓ Checks for NLTK data (downloads 'wordnet' if needed)
2. ✓ Clones LUMA repository to `external/LUMA/` (if not present)
3. ✓ Verifies raw data is available in `data/luma_raw/`
4. ✓ Creates compilation configuration
5. ✓ Organizes data into train/test splits:
   - **42 in-distribution classes** (500 train + 100 test samples each)
   - **8 out-of-distribution classes** for OOD experiments
6. ✓ Saves compiled dataset to `data/luma_compiled/`

**Compilation Configuration:**

The default configuration (in `compile_luma.py`) is:
```python
LUMA_CONFIG = {
    'data_path': 'data/luma_raw',           # Raw data location
    'output_path': 'data/luma_compiled',    # Compiled data location
    'train_samples_per_class': 500,         # Training samples per class
    'test_samples_per_class': 100,          # Test samples per class
    'num_classes': 42,                       # In-distribution classes
    'num_ood_classes': 8,                    # OOD classes
    
    # Modalities to include
    'use_audio': True,
    'use_text': True,
    'use_image': True,
    
    # No noise by default (clean dataset)
    'sample_noise': {'add_noise_train': False, 'add_noise_test': False},
    'label_noise': {'add_noise_train': False, 'ratio': 0.0},
    'ood_samples': {'add_ood': False, 'ratio': 0.0},
}
```

To customize compilation settings, edit these values in `compile_luma.py` before running.

---

### Step 3: Verify the Compilation

Test that everything compiled correctly:

```bash
python test_luma.py
```

**Expected output:**
```
============================================================
TEST 1: Checking compiled dataset files
============================================================
  ✓ data/luma_compiled/audio_datalist.csv
  ✓ data/luma_compiled/text_data.tsv
  ✓ data/luma_compiled/edm_images.pickle
  ✓ data/luma_compiled/metadata.yaml
  ✓ audio_path.txt (references: /path/to/data/luma_raw/audio)

✓ All required files present!

============================================================
TEST 2: Loading dataset
============================================================
  Creating dataloaders...
  ✓ Dataset loaded successfully
    - Classes: 42
    - Modalities: 3
    - Dimensions: [[40], [128], [3072]]
    - Train batches: 5250
    - Test batches: 1050

============================================================
TEST 3: Loading a batch
============================================================
  Batch shapes:
    - Audio: torch.Size([4, 40])
    - Text: torch.Size([4, 128])
    - Image: torch.Size([4, 3072])
    - Labels: torch.Size([4])

✓ All tests passed!
```

If all tests pass, your dataset is ready to use!

---

## Dataset Structure

After compilation, your directory structure should look like:

```
data/
├── luma_raw/                    # Raw downloaded data (in .gitignore)
│   ├── audio/
│   │   ├── datalist.csv
│   │   ├── cv_audio/           # Mozilla Common Voice
│   │   ├── sw_audio/           # Spoken Wikipedia
│   │   └── ls_audio/           # LibriSpeech
│   ├── text_data.tsv
│   └── edm_images.pickle
│
└── luma_compiled/               # Compiled dataset (in .gitignore)
    ├── audio_datalist.csv       # Audio indices and labels
    ├── audio_path.txt           # Reference to audio directory
    ├── text_data.tsv            # Text data and labels
    ├── edm_images.pickle        # Image data and labels
    └── metadata.yaml            # Dataset configuration
```

**Important:** Both `luma_raw/` and `luma_compiled/` are in `.gitignore` and won't be committed to version control.

---

## Dataset Specifications

### Modality Details

| Modality | Raw Format | Processed Format | Dimension |
|----------|------------|------------------|-----------|
| **Audio** | .wav files (varying length) | 40-dim MFCC features | (batch, 40) |
| **Text** | UTF-8 text passages | Token IDs or embeddings | (batch, 128) |
| **Image** | 32×32 RGB images | Flattened pixel values | (batch, 3072) |

### Class Distribution

- **Total classes:** 50
- **In-distribution (training):** 42 classes
  - Training samples: 500 per class (21,000 total)
  - Test samples: 100 per class (4,200 total)
- **Out-of-distribution (OOD):** 8 classes
  - Used for uncertainty quantification experiments

### Audio Sources

The audio modality combines pronunciations from three datasets:
1. **Mozilla Common Voice** (CC0 license)
2. **Spoken Wikipedia** (CC BY-SA 4.0 license)  
3. **LibriSpeech** (CC BY 4.0 license)

### Text Generation

Text passages were generated using **Gemma-7B Instruct LLM**, describing the class labels. Note: This is LLM-generated content and may contain biases or factual inaccuracies. The dataset should only be used for uncertainty quantification research.

### Image Sources

Images come from:
1. **CIFAR-10/100 datasets** (50-class subset)
2. **EDM-generated images** (from same distribution)

---

## Troubleshooting

### Issue: "git lfs" not found

**Problem:** Git LFS is not installed.

**Solution:**
```bash
# Install git-lfs
# Ubuntu/Debian:
sudo apt-get install git-lfs

# macOS:
brew install git-lfs

# Then initialize:
git lfs install
```

### Issue: "Raw LUMA data not found"

**Problem:** The raw data hasn't been downloaded.

**Solution:**
```bash
# Download from HuggingFace
git clone https://huggingface.co/datasets/bezirganyan/LUMA data/luma_raw
```

### Issue: "NLTK 'wordnet' resource not found"

**Problem:** NLTK data is missing.

**Solution:**
The compilation script auto-downloads this. If it fails:
```python
import nltk
nltk.download('wordnet')
```

### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Problem:** Missing dependencies.

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Compiled dataset already exists. Recompile? (y/N)"

**Problem:** You're running compilation again on an existing dataset.

**Solution:**
- Type `n` to use the existing compiled dataset
- Type `y` to recompile (useful if you changed configuration)

### Issue: "compile_dataset.py not found in LUMA repository"

**Problem:** The LUMA repository's compilation script is not available.

**Solution:**
The compilation script automatically falls back to a simplified compilation process. This is normal and the dataset will still work correctly.

---

## Advanced: Custom Compilation

### Injecting Uncertainty

To compile the dataset with controlled uncertainty (noise, label errors, OOD samples), edit `compile_luma.py`:

```python
LUMA_CONFIG = {
    # ... other settings ...
    
    # Add sample noise (realistic noise to each modality)
    'sample_noise': {
        'add_noise_train': True,   # Add noise to training data
        'add_noise_test': False,   # Keep test data clean
    },
    
    # Add label noise (switch labels to confuse model)
    'label_noise': {
        'add_noise_train': True,
        'ratio': 0.1,  # 10% of labels randomly switched
    },
    
    # Include OOD samples
    'ood_samples': {
        'add_ood': True,
        'ratio': 0.1,  # 10% OOD samples in dataset
    }
}
```

Then recompile:
```bash
python compile_luma.py  # Will ask to recompile
```

### Using Only Specific Modalities

To compile with only certain modalities:

```python
LUMA_CONFIG = {
    # ... other settings ...
    
    'use_audio': True,   # Include audio
    'use_text': False,   # Exclude text
    'use_image': True,   # Include image
}
```

This creates a bimodal dataset (audio + image only).

---

## Dataset Citation

If you use the LUMA dataset in your research, please cite:

```bibtex
@inproceedings{luma_dataset2025,
  title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data},
  author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

**Paper:** [arXiv:2406.09864](https://arxiv.org/abs/2406.09864)  
**Code:** [github.com/bezirganyan/LUMA](https://github.com/bezirganyan/LUMA)  
**Dataset:** [huggingface.co/datasets/bezirganyan/LUMA](https://huggingface.co/datasets/bezirganyan/LUMA)

---

## License Information

### LUMA Dataset
The LUMA compilation tool and scripts are provided under their respective licenses. Please check the LUMA repository for details.

### Audio Data
- Mozilla Common Voice: CC0 license
- Spoken Wikipedia: CC BY-SA 4.0 license
- LibriSpeech: CC BY 4.0 license

### Image Data
- CIFAR-10/100: Available for research use
- EDM-generated images: Check original dataset license

### Text Data
Generated using Gemma-7B LLM. Use is limited to research purposes for multimodal uncertainty quantification.

---

## Next Steps

After successfully compiling the LUMA dataset:

1. **Configure feature encoders** (IMPORTANT!)
   - Update `run_luma.py` to use proper encoders instead of `IdentityEncoder`
   - See `QUICK_START.md` for instructions

2. **Run experiments**
   ```bash
   python run_luma.py
   ```

3. **Analyze results**
   - Check `results/` directory for metrics
   - Compare baseline fusion vs. your DCBF approach

---

## Additional Resources

- **LUMA Paper:** [arXiv:2406.09864](https://arxiv.org/abs/2406.09864)
- **LUMA GitHub:** [github.com/bezirganyan/LUMA](https://github.com/bezirganyan/LUMA)
- **Medium Tutorial:** [Uncertainty-Aware AI from Multimodal Data](https://medium.com/@grigor.bezirganyan98/uncertainty-aware-ai-from-multimodal-data-a-pytorch-tutorial-with-luma-dataset-dfd37fc12acd)

---

## Support

If you encounter issues with dataset compilation:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check the LUMA GitHub issues: [github.com/bezirganyan/LUMA/issues](https://github.com/bezirganyan/LUMA/issues)
4. Ensure you have sufficient disk space (~15 GB total)

For issues specific to this thesis implementation, check the main README or create an issue in this repository.