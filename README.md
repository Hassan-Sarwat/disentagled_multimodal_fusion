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