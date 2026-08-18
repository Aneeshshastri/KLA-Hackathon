# Silicon Optometrists - KLA Hackathon Submission

This is the final submission package for team **Silicon Optometrists**.

## Solution Overview
Our model uses a **NAFNet (Nonlinear Activation Free Network)** architecture built with `flax.nnx` and JAX.
The model was heavily modified to include a degradation encoder and multi-scale feature trunks to handle diverse noise profiles. 
The final model is quantized to **FP8** using the `qwix` library, bringing the file size down to < 2MB while preserving high fidelity restoration.

## Requirements
- Python 3.10+
- NVIDIA GPU

## Setup Instructions
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Note: JAX will install the CUDA-compatible runtime automatically via `jax[cuda12]`. If you have different CUDA drivers, refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).)*

## Execution
To run the model on a directory of noisy `.npy` files:
```bash
python run.py <input-dir> <output-dir>
```

**Constraints Met**:
- ✅ Generates one `.npy` file per input file with matching filenames.
- ✅ Outputs are grayscale `(H, W)` arrays within the `[0, 1]` range (no NaNs or Infs).
- ✅ Output directory is automatically created if it doesn't exist.
- ✅ Fully offline, no external downloads or API keys needed.
- ✅ Packaged model weights are fully self-contained.
