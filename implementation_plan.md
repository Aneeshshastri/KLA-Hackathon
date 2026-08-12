# Proposal 2: Mixture of Experts (MoE) for Iterative Denoising

Based on the problem statement details and the degradation script, there are 4 types of degradations applied to the images:
1. **Gaussian Blur**
2. **Speckle Noise**
3. **Additive Gaussian Noise**
4. **Downsampling**

This proposal focuses on separating the denoising process into specialized experts, each pre-trained to handle a single type of degradation. These experts will be orchestrated by a router network over a 4-step iterative inference loop to reconstruct the final image.

## Open Questions

> [!WARNING]
> Before proceeding with execution, I need clarification on a few points:
> 1. **Fourth Expert:** You mentioned "the 4th expert is really to apply downsampling". Do you mean this expert should be pre-trained to perform **upsampling / super-resolution** (the inverse of downsampling) so that it restores the resolution? Or do you mean it actually applies the forward downsampling operation (e.g., as part of an algorithm-unrolled data consistency step)? Assuming you mean an expert for upsampling to invert the downsampling.
> 2. **Pre-training:** Should we generate the isolated datasets and write the pre-training scripts as part of this proposal execution, or do you have pre-trained experts ready? 
> 3. **MoE Routing:** Should the MoE router output a weighted blend of all experts (Soft Routing) at each step, or select exactly one expert (Hard Routing/Gumbel Softmax)? 

## Proposed Changes

We will create a new directory `Proposal_2` to contain this approach.

### 1. Data Generation (Proposal_2/generate_data.sh)
- Script to run `noise_reconstruction_generator.py` multiple times to generate isolated datasets:
  - `data/train/blur_only`
  - `data/train/gaussian_only`
  - `data/train/speckle_only`
  - `data/train/downsample_only`

### 2. Expert Models (Proposal_2/moe_experts.py)
#### [NEW] Proposal_2/moe_experts.py
- Refactor the existing `BaselineNAFNet` to serve as a standalone expert.
- Create 4 expert instances for the 4 specific degradations. The upsampling expert will specifically handle restoring spatial resolution.

### 3. MoE Router & Iterative Pipeline (Proposal_2/moe_model.py)
#### [NEW] Proposal_2/moe_model.py
- **Router Network**: Uses a Convolutional encoder that outputs a distribution over the available experts.
- **IterativeMoEPipeline**: 
  - Takes the degraded image `x_0`.
  - Runs a `for i in range(4)` loop.
  - In each step, the router evaluates `x_i`, computes expert weights, and generates `x_{i+1}` as a weighted sum of the experts' outputs. 
  - *Note: Since the upsampling expert changes the image dimensions, the router and subsequent experts will need to handle variable spatial resolutions, or we apply the upsampling expert exactly once at the correct step.*

### 4. Training and Evaluation Scripts
#### [NEW] Proposal_2/train_moe.py
- Script to train the individual experts.
- Script to freeze the experts and train *only* the MoE Router network end-to-end on the combined noisy dataset.
#### [NEW] Proposal_2/eval_moe.py
- Uses `evaluator.py` to evaluate the 4-step MoE pipeline.

## Verification Plan

### Automated Tests
- Run `python Proposal_2/moe_model.py` to verify that the MoE pipeline can successfully initialize and execute a 4-step forward pass without shape mismatches (especially important around the resolution change).
- Ensure the router outputs valid weights across the 4 steps.

### Manual Verification
- Review the generated dataset manifests to ensure the isolated noise conditions are correct.
- Evaluate the MoE Router's behavior.
