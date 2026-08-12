import flax.nnx as nnx
import jax.numpy as jnp, jax.image as jimg, jax
import lpips_jax
import numpy as np

import os

import matplotlib.pyplot as plt

# ==============================================================================
#                      Metric Definitions (+ misc. defs)                             
# ==============================================================================

lpips_alex = lpips_jax.LPIPSEvaluator(net='alexnet', replicate=False)

def LPIPS(gt:jnp.array, im:jnp.array):
    gt = 2*jnp.concatenate([gt] * 3, axis=-1)-1
    im = 2*jnp.concatenate([im] * 3, axis=-1)-1
    distance = lpips_alex(gt, im)
    return jnp.reshape(distance, (-1, 1))

def SSIM(gt:jnp.array, im:jnp.array):
    mean_gt = jnp.mean(gt, axis=[1,2])
    mean_im = jnp.mean(im, axis=[1,2])
    var_gt = jnp.var(gt, axis=[1,2])
    var_im = jnp.var(im, axis=[1,2])
    cross_var = jnp.mean(gt*im, axis=[1,2]) - mean_gt * mean_im
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim = (2*mean_gt*mean_im + c1)*(2*cross_var + c2)/((mean_gt**2 + mean_im**2 + c1)*(var_gt + var_im + c2))
    return ssim

def PSNR(gt:jnp.array, im:jnp.array):
    max_pixel = 1.0
    mse = jnp.mean((gt-im)**2, axis=[1,2])
    psnr = 20 * jnp.log10(max_pixel/(jnp.sqrt(mse) + 1e-8))
    return psnr

@nnx.jit
def forward(model, x):
    return model(x)


#———————————————————————— Model Evaluation Class —————————————————————————                        
# ModelEvaluator.validate()  -->  to be used in a single validation step
# ModelEvaluator.evaluate() -->  large scale calculation of metrics from 
#                                 testing/evaluation purposes
class ModelEvaluator:
    def __init__(self, gt_path=None, noisylr_path=None, mpi_path=None, n_eval=None):
        self.metrics = {'SSIM':SSIM, 'PSNR':PSNR, 'LPIPS':LPIPS}
        self.data = {}
        self.gt_path = gt_path
        self.noisylr_path = noisylr_path
        self.mpi_path = mpi_path
        self.n_eval = n_eval
        if n_eval==None:
            self.n_eval = len(os.listdir(gt_path))

    def validate(self, pred, gt) -> jax.Array: 
        # Get evaluation metrics at each validation step
        # INPUT:    pred - Reconstructed Images, in the shape (B, H, W, C)
        #           gt   - Ground Truth Images,  in the shape (B, H, W, C)
        # OUTPUT:   jax.Array([metric1_mean, metric2_mean, ...])

        vals = []
        for metric, metric_fn in self.metrics.items():
            vals.append(metric_fn(gt, pred).mean())
        return jnp.stack(vals)


    def evaluate(self, model):
        # Gets evaluation metrics for each GT-Reconstructed pair in testing split, and stores them in self.data
        assert self.gt_path!=None, "Please provide a ground-truth image path directory!"
        assert self.noisylr_path!=None, "Please provide a noisy low-resolution image path directory!"
        assert self.mpi_path!=None, "Please provide a path directory for storing model restored images!"

        model.eval()
        
        for metric in self.metrics.keys():
            self.data[metric] = []

        for i in range(self.n_eval):
            gt = jnp.concatenate([jnp.expand_dims(jnp.load(self.gt_path+f'{i:06d}.npy'),0)[:,:,:,jnp.newaxis]]*2,axis=0)
            noisylr = jnp.concatenate([jnp.expand_dims(jnp.load(self.noisylr_path+f'{i:06d}.npy'),0)[:,:,:,jnp.newaxis]]*2,axis=0)

            mpi = forward(model, jnp.expand_dims(noisylr, axis=0)) 

            jnp.save(self.mpi_path + f'{i:06d}.npy', mpi)

            for metric in self.metrics.keys():
                if self.metrics[metric]==None:
                    continue
                cur = self.metrics[metric](gt, mpi)
                self.data[metric].append(cur)

        

    def load_pair(self, idx):
            gt = np.asarray(jnp.load(self.gt_path + f'{idx:06d}.npy'))
            mpi = np.asarray(jnp.load(self.mpi_path + f'{idx:06d}.npy'))
            return gt, mpi


    def display(self, n_random=3, n_extreme=3, higher_is_better=None):
        assert self.gt_path!=None, "Please provide a ground-truth image path directory!"
        assert self.noisylr_path!=None, "Please provide a noisy low-resolution image path directory!"
        assert self.mpi_path!=None, "Please provide a path directory for storing model restored images!"
        
        if higher_is_better is None:
            higher_is_better = {'SSIM': True, 'PSNR': True, 'LPIPS': False}
        
        random_idxs = np.random.choice(self.n_eval, size=n_random, replace=False)
        fig, axes = plt.subplots(2, n_random, figsize=(3 * n_random, 6))
        for col, idx in enumerate(random_idxs):
            gt, mpi = self.load_pair(idx)
            axes[0, col].imshow(gt, cmap='gray')
            axes[0, col].axis('off')
            axes[0, col].set_title(f"idx {idx}", fontsize=9)
            axes[1, col].imshow(mpi, cmap='gray')
            axes[1, col].axis('off')
        axes[0, 0].set_ylabel("GT", fontsize=12)
        axes[1, 0].set_ylabel("Recon", fontsize=12)
        fig.suptitle("Random samples")
        plt.tight_layout()
        plt.show()

        for metric in self.metrics.keys():
            if self.metrics[metric] is None:
                continue

            scores = np.asarray(self.data[metric])
            hib = higher_is_better.get(metric, True)
            order = np.argsort(scores)

            best_idxs = order[-n_extreme:][::-1] if hib else order[:n_extreme]
            worst_idxs = order[:n_extreme] if hib else order[-n_extreme:][::-1]

            fig, axes = plt.subplots(2, n_extreme * 2, figsize=(3 * n_extreme * 2, 6))
            for i, idx in enumerate(best_idxs):
                gt, mpi = self.load_pair(idx)
                axes[0, i].imshow(gt, cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title(f"Best {metric}\nidx {idx}, {scores[idx]:.4f}", fontsize=9)
                axes[1, i].imshow(mpi, cmap='gray')
                axes[1, i].axis('off')

            for i, idx in enumerate(worst_idxs):
                col = n_extreme + i
                gt, mpi = self.load_pair(idx)
                axes[0, col].imshow(gt, cmap='gray')
                axes[0, col].axis('off')
                axes[0, col].set_title(f"Worst {metric}\nidx {idx}, {scores[idx]:.4f}", fontsize=9)
                axes[1, col].imshow(mpi, cmap='gray')
                axes[1, col].axis('off')

            axes[0, 0].set_ylabel("GT", fontsize=12)
            axes[1, 0].set_ylabel("Recon", fontsize=12)
            fig.suptitle(metric)
            plt.tight_layout()
            plt.show()