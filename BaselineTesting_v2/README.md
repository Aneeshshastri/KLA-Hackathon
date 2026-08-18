# Baseline Testing V2

### Edition 1
<img width="270" height="470" alt="image" src="https://github.com/user-attachments/assets/dca480c5-b57d-493b-a5fe-0f8d0a5b77a1" />  

Our first baseline follows the above architecture.
1. Up-convolution results in batched images of shape (B, W/upscale, H/upscale, C * upscale^2), which is then corrected by depth-to-space up sampling by Pixel Shuffle to (B, W, H, C). This was done, as directly upscaling to (..., H, W, ...) would have introduced a very abrupt increase in information generated, which may cause poor performance.
2. A residual connection was created for the output, combining Pixel Shuffled outputs and Bicubic Upscaled inputs.

It poorly performed in properly denoising, particularly speckle noise. It was unable to adequately deblur images. Another observed failure was the lack of texture in model outputs — or the lack thereof.

### Edition 2
We were motivated to try and improve the model by curating a "Degradation Encoder" — a model that converted a noisy image into an embedding that indicated how degraded it was, and also information about the type of degradation.
This was considered as we assumed that each NAFNet Block in Edition-1 required some kind of transformation that changed based on how degraded the previous output was.

In our exact implementation, our Degradation Encoder acted as illustrated above.
We conditioned each NAFNet Block on the resulting "Degradation Embedding" using Feature-Wise Linear Modulation (FiLM)
The resulting bias and scale are used to condition the output of the previous NAFNet Block before it is fed to the next; thereby making each block aware of how degraded the passed input actually is.

Due to the worse metrics, we looked into what was potentially hindering this setup.

### Edition 3
DWT — Discrete Wavelet Transform — helps to separate the different frequency band features in an image — LL, HL, LH, HH — all capturing different natures of the image. LL captures the coarse, low-frequency features of the image — like blurs, LH covers vertical edges and horizontal features, HL covers horizontal edges and vertical features, and HH captures diagonal edges and high frequency features like noise.

This separation of low and high frequency components and how they affect noises and blurs acted like a signature, that could be used to better help in model learning.

In our implementation, we performed Haar DWT on the normalised input. The Non-LL components were fed into a `DetailFusion` module, that combined them via convolutional layers.
The LL component was also fed into a convolutional extraction layer, with its output and the DetailFusion output being summed together.
The resultant was then directly taken through a sequence of NAFNet Blocks w/ FiLM conditioning.
*Pixel shuffling was only done by a factor of x2 to compensate for the reduction in H and W by Haar DWT.

### Edition 4
To try and improve the restored images, we implemented an `AdaptiveFrequencyBlock`.
It is a module that learns projections of low and high frequency components of the image, derived from FFT. The high frequency projections, representing noise and other variations, are then gated by the low frequency — thus bringing them down and smoothing the inputs, before it goes into the NAFNet Trunk.

While there are some improvements, we thought of a more novel approach to dealing with the problem.

### Edition 5
A 3-expert MoE model was implemented, that had single CNN layer heads, each specialised via a `FrequencyRouter` that assigned soft routing weights based on the input image's frequency content. Rather than a single shared output projection, the trunk's features were passed through three separate expert heads (each a single conv + pixel shuffle), and their outputs were combined via a weighted sum, with weights predicted directly from the raw input.

The intuition here was that a single output head, regardless of how well-conditioned the trunk was, still had to compress its transformation into one shared function — whereas separate expert heads could specialise toward different degradation regimes (e.g. one leaning toward denoising-style smoothing, another toward detail reconstruction for downsampled inputs), with the router learning to blend them appropriately per image.

### Edition 6
Edition 6 kept the same 3-expert MoE routing setup from Edition 5, but replaced the plain `DegradationEncoder` with the more elaborate `BlindDFCTokenEncoder` for producing the FiLM conditioning embedding fed to the trunk.

Instead of a simple conv-conv-project encoding of the raw image, this encoder operated in the frequency domain directly — it computed the FFT magnitude spectrum of the input, derived a radial energy profile, and used a small learned network to predict adaptive band centres and bandwidths for splitting the spectrum into a set of frequency bands. Each band was passed through its own conv extractor and projected into a token, and the resulting per-band tokens were combined via a learned soft-routing/attention step into a single degradation embedding.

The motivation was that a coarse spatial encoding (as in Edition 2/5) may not capture *what kind* of degradation is present as precisely as an encoding derived directly from the image's frequency signature — noises captured by the high frequency components, and blurs captured by low frequency components. Indirectly, we hoped this would lead to better informed routing by the `FrequencyRouter`.

---

### Summary of Results

| Edition | Description                                 | Mean SSIM | Mean PSNR | Mean LPIPS | Sample                                    |
| ------- | ------------------------------------------- | --------- | --------- | ---------- | ----------------------------------------- |
| 1       | Baseline (Pixel Shuffle + bicubic residual) | 0.7298    | 27.6306   | 0.3170     | ![Generated_Samples1](BaselineTesting_v2/results/E1_ZeroDegradationEnc/output_train.png) |
| 2       | + Degradation Encoder + FiLM                | 0.7276    | 27.5651   | 0.3216     | ![Generated_Samples1](BaselineTesting_v2/results/E2_DegradationEnc_Residual/output_train.png) |
| 3       | + Haar DWT + DetailFusion                   | 0.7149    | 27.2994   | 0.3447     | ![Generated_Samples1](BaselineTesting_v2/results/E3_Wavelet_DirectRecon/output_train.png) |
| 4       | + AdaptiveFrequencyBlock                    | 0.7279    | 27.5763   | 0.3206     | ![Generated_Samples1](BaselineTesting_v2/results/E4_AdaIR_residual/output_train.png) |
| 5       | + 3-expert MoE (FrequencyRouter)            | 0.7337    | 27.7419   | 0.3050     | ![Generated_Samples1](BaselineTesting_v2/results/E5_3RoutedResdiualHeads/output_train.png) |
| 6       | + BlindDFCTokenEncoder                      | 0.7353    | 27.7940   | 0.2974     | ![Generated_Samples1](BaselineTesting_v2/results/E6_DFCToken_3ResdiualHeads/output_train.png) |

---

### Precautions and Limitations
1. Only the original 3,200 images were used initially, of which 2,800 were actually used for training, with the balance going towards validation. Synthetic data generation was implemented after the baselines.
2. Model capacities were kept minimal for ablation purposes and due to the smaller training data size, limiting how much each baseline could learn.
