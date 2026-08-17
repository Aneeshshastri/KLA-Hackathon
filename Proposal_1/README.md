# Proposal 1: NAFNet Baseline with Pre-trained Weight Porting

This directory contains the first proposal for the KLA-Hackathon image restoration task. It serves as a strong baseline by leveraging **NAFNet** (Nonlinear Activation Free Network), a state-of-the-art image restoration architecture, implemented in JAX/Flax. 

To jumpstart performance, this proposal includes a complete pipeline to port pre-trained PyTorch weights (trained on the SIDD dataset) into JAX, followed by an architecture modification to fine-tune it for the specific 1-channel to 1-channel upsampling task.

## 📁 Repository Structure

- **`nafnet_pytorch.py`**: The original PyTorch implementation of NAFNet. Used as a reference for architectural parity.
- **`nafnet_jax.py`**: The JAX/Flax (`nnx`) port of the standard NAFNet architecture. It implements core components like `SimpleGate`, `NAFBlock`, and `StandardNAFNet`.
- **`port_weights.py`**: A utility script that downloads the pre-trained PyTorch weights (`NAFNet-SIDD-width64.pth`) from Hugging Face, maps the PyTorch state dict keys to their corresponding JAX/Flax `nnx` parameters, handles tensor transpositions (e.g., Conv kernels from `[out_c, in_c, kH, kW]` to `[kH, kW, in_c, out_c]`), and saves the resulting state dict as `nafnet_pretrained.msgpack`.
- **`finetune_pipeline.py`**: The wrapper module (`RestorationPipeline_Finetune`) that adapts the pre-trained model for our specific task.
- **`Proposal_1.ipynb`**: A Jupyter Notebook demonstrating the usage, training loop, or evaluation of this fine-tuning pipeline.
- **`jax_keys.txt`**: A reference text file containing the mapped state dict keys for the JAX model.

## 🧠 Architecture & Fine-Tuning Strategy

The pre-trained NAFNet was originally trained to denoise 3-channel RGB images at a constant resolution. However, our task requires taking a 1-channel (grayscale) noisy low-resolution image and outputting a 1-channel clean high-resolution image. 

`finetune_pipeline.py` handles this adaptation in four steps:
1. **Input Adaptation**: The 1-channel grayscale input is repeated across the channel dimension to create a pseudo-3-channel image.
2. **Pre-trained Backbone**: The image is passed through the pre-trained `StandardNAFNet` backbone, which operates on the 3-channel input and outputs a denoised 3-channel feature map.
3. **Untrained Upsample Tail**: The features are passed through an `UpsampleTail`, consisting of an initial projection, two `NAFBlock`s, and a `PixelShuffle` layer to achieve a 2x spatial upsampling.
4. **Final Projection**: A final convolution projects the upsampled 64-channel feature map down to the target 1-channel high-resolution output.

## 🚀 Getting Started

### 1. Porting the Pre-trained Weights
Before training or evaluating, you need to download and port the PyTorch weights to JAX. Run the following script:
```bash
python port_weights.py
```
This will:
- Download `NAFNet-SIDD-width64.pth` (if not already present).
- Instantiate the JAX model and map the weights.
- Save the JAX-compatible weights to `nafnet_pretrained.msgpack`.

### 2. Fine-Tuning the Model
Once the weights are ported, the model can be fine-tuned. The `load_pretrained_nafnet` function inside `finetune_pipeline.py` is used to load `nafnet_pretrained.msgpack` directly into the `RestorationPipeline_Finetune`'s backbone. You can follow `Proposal_1.ipynb` to see the complete data loading and training loop.
