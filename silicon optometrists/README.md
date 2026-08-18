# Silicon Optometrists - KLA Hackathon Submission

This is the final submission package for team **Silicon Optometrists**.

## Solution Overview
Our model uses a **NAFNet (Nonlinear Activation Free Network)** architecture built with `flax.nnx` and JAX.
The model was heavily modified to include a degradation encoder and multi-scale feature trunks to handle diverse noise profiles. 
The final model is quantized to **FP8** using the `qwix` library, bringing the file size down to < 4MB while preserving high fidelity restoration.

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

## Motivations

In the task of *Image Restoration*, especially for semiconductor die inspections, degradations and distortions in the observed image feed make it difficult to correctly judge wafer surface/die defects. This in turn results in false rejects, or even worse, defective dies that make their way into packaging, resulting in catastrophic failures down the line.
To mitigate this, the *Image Restoration* model-
1. needs to be invariant to the order of degradations applied, since real degradation pipelines don't distort images in any fixed sequence
2. must not introduce visual artifacts that could obscure or mimic actual defects
3. needs speedy inference for real-time fault inspection

To earnestly tackle these needs, we explored all three aspects of model learning-
1. **Data generation** - Including  augmentations and synthetic data generation efforts for increasing the training data size.
2. **Model architecture** - Detailing on our NAFNet-based backbone, with several modifications integrated to fit the task.
3. **Model training** - Our loss functions and other optimisation strategies.

## Data Generation 
### Synthetic Data Generation 
By leveraging Discrete Frequency Curves (DFCs), we can sample Gaussian blur variance from a pre-determined range, and by comparing their radial-binned profiles to those of the ground truth - letting us compare the exact low frequency profile that blurs are detectable at, letting us estimate the Gaussian blur parameters.
Next, we bin the residual of GT and Noisy images, and fit a line for Variance vs Bin Intensity. 
The slope gives us our Speckle (multiplicative) noise parameters, while the intercept gives the same for the Gaussian (additive) noise.
We repeat this process over the entire original Ground Truth dataset, and build a distribution of noise and blur parameters. We sample from this, and generate the desired number of synthetic data points.

### Data Augmentation
On the fly augmentation, by rotating and flipping.

## Model architecture
<img width="730" height="990" alt="image" src="https://github.com/user-attachments/assets/2761b19a-8688-4092-bcb7-fc140f02f53f" />  

In our model architecture, the backbone is powered by a FiLM-conditioned, dilated NAFNet trunk, conditioned on a "degradation embedding" using Feature-Wise Linear Modulation, which adjusts the inputs by predicting scaling and biases for them. Atrous convolutional layers in the NAFNet trunk allows the model to have a greater field of view, potentially helping with spread out effects like blur. Beyond the dilated NAFNet trunk, the trunk also incorporates multi-scale feature processing; shallow blocks feed into strided downsampling, a bottleneck, and skip fusion; allowing the model to reason over multiple spatial scales in a single pass.

This degradation embedding is produced by the Adaptive FFT Band Token Encoder. It computes the FFT magnitude spectrum of the input, derives a radial energy profile, and uses a small learned network to predict adaptive band centres and bandwidths for splitting the spectrum into a set of frequency bands. Each band is then passed through its own conv extractor and projected into a token; these per-band tokens are combined via a learned soft-routing step into the final degradation embedding.

The soft image-level router produces routing weights that combine with the trunk's processed features at three residual experts; *three specialized heads* producing candidate residuals, blended into a single weighted residual.

This weighted residual is then added to a bicubic up-sampled NoisyLR input, before pixel intensities are clamped to the range [0, 1]. A straight-through estimator ensures gradients still flow through the clamp for out-of-range predictions, so the model continues learning to correct them rather than stalling at the boundary due to signals turning silent.
## Training Losses and Optimisations
We used the following losses:

The model is trained on a weighted combination of six losses, each targeting a different aspect of restoration quality:
**Charbonnier Loss** — a smooth approximation of L1 (sqrt((pred - target)² + ε²)), used as the primary pixel-space reconstruction loss. Retains L1's robustness to outliers while staying differentiable near zero.

**FFT Loss** — L1/L2 distance between the FFT magnitudes of prediction and target, penalising frequency-domain discrepancies directly rather than relying on pixel-space error alone.

**Flat Residual Variance Loss** — penalises residual variance in flat/textureless regions, discouraging leftover speckle/Gaussian noise in areas with no true texture to explain it.

**SSIM Loss** — 1 - SSIM(pred, target), encouraging structural and perceptual fidelity over raw pixel matching.

**Eagle Loss** — a frequency-domain loss targeting high-frequency detail recovery, counteracting the over-smoothing tendency of pixel/structural losses.

**LPIPS Loss** — perceptual distance measured in a pretrained network's feature space, correlating more closely with human visual judgment than pixel-space losses.

We weighted the losses as (0.55, 0.15, 0.05, 0.15, 0.05, 0.05).

## Qualitative Examples
Here are some comparisons - 
			Low Resolution input VS Model output VS Ground truth
<img width="583" height="616" alt="image" src="https://github.com/user-attachments/assets/c5324b8e-b527-441c-8005-bda6d2cd9569" />
<img width="588" height="592" alt="image" src="https://github.com/user-attachments/assets/90ab7efc-109f-4861-9711-66716abf36b9" />

## Optimisations
We utilised meshing to make use of TPUs for training. Grain for dataloaders.

The final model is quantized to **FP8** using the `qwix` library, bringing the file size down to < 2MB while preserving high fidelity restoration. This gave a significant inference speed boost, while also taking up lesser VRAM.
