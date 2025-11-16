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

Your dataset class should be structured as follows. You can add it in a new file in datasets folder or in `datasets/dataset.py`.

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

## Adding a New Disentanglement Method

To add a new disentanglement method, you need to create a new PyTorch Lightning module and integrate it into the training and evaluation pipeline in `run.py`.

### 1. Create a New Model Class

Create a new Python file in the `models/` directory (e.g., `models/my_disentangler.py`). In this file, define your model as a class that inherits from `pytorch_lightning.LightningModule`.

Your class should look something like this:

```python
# in models/my_disentangler.py
import pytorch_lightning as pl
import torch
import torch.nn as nn

class MyDisentangler(pl.LightningModule):
    def __init__(self, input_dims, embed_dim, ...):
        super().__init__()
        self.save_hyperparameters()

        # Define your model's layers (encoders, decoders, etc.) here
        self.encoders = nn.ModuleList(...)
        self.decoders = nn.ModuleList(...)
        # ...

    def forward(self, x_list):
        # Implement the forward pass of your model
        # This should include encoding, disentanglement, and reconstruction
        # ...

        # Return the total loss and a dictionary of logs
        loss = ...
        logs = {'loss': loss, ...}
        return loss, logs

    def training_step(self, batch, batch_idx):
        # Standard training step
        xs = [b.float() for b in batch[:-1]]
        loss, logs = self(xs)
        self.log_dict({f'train/{k}': v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        # Define your optimizer and scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def get_embedding(self, x_list):
        # This method is crucial for evaluation.
        # It should return the disentangled representations (embeddings).
        # For example, return shared and private embeddings.
        shared_embedding = ...
        private_embeddings = [...]
        return shared_embedding, private_embeddings
```

**Key Requirements:**

-   Your model **must** be a `pl.LightningModule`.
-   The `forward` method should compute and return the loss and a log dictionary.
-   You **must** implement a `get_embedding` method that returns the learned disentangled representations. This is used by the downstream evaluation "probe" models.

### 2. Integrate into `run.py`

Now, you need to integrate your new model into the main experiment script, `run.py`.

1.  **Import your new model:**

    ```python
    # In run.py
    from models.my_disentangler import MyDisentangler
    ```

2.  **Create a factory for your model:**

    In the `build_factories` function, add a factory for your new model. This allows you to configure it from the `config.yaml` file.

    ```python
    # In run.py's build_factories function
    MyDisentanglerFactory = partial(
        MyDisentangler,
        input_dims=model_params["output_dims"],
        # ... other parameters for your model
    )
    # return it along with the others
    return DMVAEFactory, ProbeFactory, DisProbeFactory, LateFusionFactory, MyDisentanglerFactory
    ```

3.  **Instantiate and train your model:**

    In the main loop of `run.py`, instantiate your model using its factory and train it with a `pl.Trainer`.

    ```python
    # In run.py's main loop
    # ...
    my_model = MyDisentanglerFactory()

    trainer = pl.Trainer(
        # ... trainer arguments
    )
    trainer.fit(my_model, train_dataloaders=train_loader)
    ```

### 3. Test and Evaluate

In this project, "testing" a disentanglement method means evaluating the quality of its learned representations on a downstream task. The `EvidentialProbeModule` and `DisentangledEvidentialProbeModule` are used for this purpose.

After your model is trained, pass the trained instance to the probe factories to create evaluation models.

```python
# In run.py, after your model is trained
dis_my_model = DisProbeFactory(my_model)
cml_my_model = ProbeFactory(my_model, aggregation="cml")
# ... and so on

# Then, train and test these probe models
# The evaluation results will be saved automatically
```

The `run.py` script is already set up to train these probe models and collect the results. The performance of the probe models on the classification task serves as the evaluation for your new disentanglement method. The results will be saved in the `logs/` directory.