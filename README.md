# Disentangled Multimodal Fusion

This project focuses on disentangled multimodal fusion for uncertainty quantification.

## Running the Code

### Uncertainty Quantification Datasets

To run the experiments on the uncertainty quantification datasets, use the following command:

```bash
python run.py
```

### Synthetic Datasets

To run the experiments on the synthetic datasets, use the following command:

```bash
python run_synthetic.py
```

## Configuration

The parameters for the experiments can be found in the `configs` directory.

- For the uncertainty quantification datasets, edit `configs/config.yaml`.
- For the synthetic datasets, edit `configs/synthetic_config.yaml`.

In `configs/synthetic_config.yaml`, you can adjust the dependencies and their percentages under the `experiment` section.

## Adding a New Dataset

To add a new dataset to the project, you need to create a custom dataset class that inherits from `torch.utils.data.Dataset` and then integrate it into the data loading pipeline.

### 1. Create a Dataset Class

Your dataset class should be structured as follows. You can add it in a new file or in `dataset.py`.

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class MyNewDataset(Dataset):
    def __init__(self, ...):
        # Load your data here.
        # For example, load from .npy or .csv files.
        # Let's assume you have two modalities (X1, X2) and labels (y)
        self.X1 = ... # Your data for modality 1
        self.X2 = ... # Your data for modality 2
        self.y = ...  # Your labels

        # The dataset class must have these attributes
        self.num_classes = len(np.unique(self.y))
        self.num_views = 2 # Number of modalities
        self.dims = [self.X1.shape[1], self.X2.shape[1]] # Feature dimensions of each modality

    def __len__(self):
        # Return the total number of samples
        return len(self.y)

    def __getitem__(self, idx):
        # Return a tuple of modalities and the label for a given index
        return self.X1[idx], self.X2[idx], self.y[idx]
```

**Important:**
- The `__getitem__` method **must** return a tuple where the last element is the label.
- The class **must** have `num_classes`, `num_views`, and `dims` as attributes.

### 2. Integrate into `run.py`

After creating your dataset class, you need to add it to the `_get_dataset` function in `run.py`:

```python
# In run.py

# If you added your dataset class in a new file, import it first
# from my_dataset_file import MyNewDataset

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
    elif dataset_name == "MyNewDataset": # Add your dataset here
        return MyNewDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

### 3. Update Configuration

Finally, add the name of your new dataset to the `normal_datasets` or `conflict_datasets` list in `configs/config.yaml` to use it in experiments.

```yaml
# In configs/config.yaml
experiment:
  normal_datasets: ['CUB', 'CalTech', 'HandWritten', 'PIE', 'Scene', 'MyNewDataset']
  # ...
```