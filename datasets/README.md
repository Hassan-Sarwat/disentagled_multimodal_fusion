# LUMA Dataset Setup

This project uses the [LUMA (Learning from Uncertain and Multimodal Data)](https://github.com/bezirganyan/LUMA) benchmark dataset for multimodal uncertainty quantification experiments.

## Quick Setup

### Prerequisites

Install Git LFS for downloading the dataset:

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
choco install git-lfs

# Initialize git-lfs
git lfs install
```

### Dataset Installation

```bash
# 1. Download raw LUMA data from HuggingFace (~5-10 GB)
git clone https://huggingface.co/datasets/bezirganyan/LUMA data/luma_raw

# 2. Compile the dataset
python compile_luma.py

# 3. Verify compilation
python test_luma.py
```

**Expected output from verification:**
```
✓ All required files present!
✓ Dataset loaded successfully
  - Classes: 42
  - Modalities: 3 (audio, text, image)
  - Dimensions: [40, 128, 3072]
```

### Dataset Structure

After compilation:
```
data/
├── luma_raw/          # Raw data from HuggingFace (in .gitignore)
└── luma_compiled/     # Processed dataset (in .gitignore)
```

**Note:** Both directories are in `.gitignore` and won't be committed. Each user downloads their own copy.

## Modality Details

| Modality | Input Dimension | Format |
|----------|----------------|--------|
| Audio | 40 | MFCC features |
| Text | 128 | Token IDs/embeddings |
| Image | 3,072 | 32×32 RGB pixels (flattened) |

## Troubleshooting

**"git lfs not found"**
```bash
# Install git-lfs (see Prerequisites above)
git lfs install
```

**"Raw LUMA data not found"**
```bash
# Download from HuggingFace
git clone https://huggingface.co/datasets/bezirganyan/LUMA data/luma_raw
```

**"Compilation fails"**
- Ensure you have all dependencies: `pip install -r requirements.txt`
- Check you have ~15 GB free disk space
- The script will auto-download NLTK data if needed

## Dataset Information

- **Classes:** 42 in-distribution + 8 out-of-distribution
- **Samples:** 500 train + 100 test per class
- **Modalities:** Audio (3 sources), Text (LLM-generated), Images (CIFAR-10/100 + EDM)
- **Total size:** ~5-10 GB (raw) + ~2-3 GB (compiled)

## Citation

If you use the LUMA dataset, please cite:

```bibtex
@inproceedings{luma_dataset2025,
  title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data},
  author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

## Further Information

For detailed information about the dataset, uncertainty injection, and baseline methods, see:
- **LUMA Repository:** [github.com/bezirganyan/LUMA](https://github.com/bezirganyan/LUMA)
- **Paper:** [arXiv:2406.09864](https://arxiv.org/abs/2406.09864)
- **Dataset:** [huggingface.co/datasets/bezirganyan/LUMA](https://huggingface.co/datasets/bezirganyan/LUMA)
- **Tutorial:** [Medium Article](https://medium.com/@grigor.bezirganyan98/uncertainty-aware-ai-from-multimodal-data-a-pytorch-tutorial-with-luma-dataset-dfd37fc12acd)