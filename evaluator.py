import flax.nnx as nnx
import jax.numpy as jnp, jax.image as jimg
import lpips_jax
import numpy as np

import os

import matplotlib.pyplot as plt


lpips_alex = lpips_jax.LPIPSEvaluator(net='alexnet', replicate=False)

def LPIPS(gt:jnp.array, im:jnp.array):
    gt = jimg.resize(gt, img_size, 'bicubic')  ## Debugging purpose
    im = jimg.resize(im, img_size, 'bicubic')  ## Debugging purpose
    gt = 2*jnp.expand_dims(jnp.stack([gt] * 3, axis=-1), 0)-1
    im = 2*jnp.expand_dims(jnp.stack([im] * 3, axis=-1), 0)-1
    distance = lpips_alex(gt, im)
    return float(distance.item())

def SSIM(gt:jnp.array, im:jnp.array):
    gt = jimg.resize(gt, img_size, 'bicubic')  ## Debugging purpose
    im = jimg.resize(im, img_size, 'bicubic')  ## Debugging purpose
    mean_gt = jnp.mean(gt)
    mean_im = jnp.mean(im)
    var_gt = jnp.var(gt)
    var_im = jnp.var(im)
    cross_var = jnp.mean(gt*im) - jnp.mean(gt)*jnp.mean(im)
    ssim = (2*mean_gt*mean_im + 1e-7)*(2*cross_var + 1e-7)/((mean_gt**2 + mean_im**2 + 1e-7)*(var_gt**2 + var_im**2 + 1e-7))
    return float(ssim)

def PSNR(gt:jnp.array, im:jnp.array):
    gt = jimg.resize(gt, img_size, 'bicubic')  ## Debugging purpose
    im = jimg.resize(im, img_size, 'bicubic')  ## Debugging purpose
    max_pixel = 1
    mse = jnp.mean((gt-im)**2)
    psnr = 20 * jnp.log10(max_pixel/jnp.sqrt(mse))
    return float(psnr)

@nnx.jit
def forward(model, x):
    return model(x)

metrics = {'SSIM':SSIM, 'PSNR':PSNR, 'LPIPS':LPIPS}

class ModelEvaluator:
    def __init__(self, model:nnx.Module, metrics:dict, gt_path, noisylr_path, mpi_path, n_eval=None):
        self.model = model
        self.metrics = metrics
        self.data = {}
        self.gt_path = gt_path
        self.noisylr_path = noisylr_path
        self.mpi_path = mpi_path
        self.n_eval = n_eval
        if n_eval==None:
            self.n_eval = len(os.listdir(gt_path))

    def evalulate(self):
        # self.model.eval()
        
        for metric in self.metrics.keys():
            self.data[metric] = []

        for i in range(self.n_eval):
            gt = jnp.load(self.gt_path+f'{i:06d}.npy')
            noisylr = jnp.load(self.noisylr_path+f'{i:06d}.npy')

            mpi = forward(self.model, jnp.expand_dims(noisylr, axis=0)) 

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


    def summary(self):
        pass


if __name__=='__main__':
    
    img_size = (256,256)

    gt_path = './dummy/GT/'
    noisylr_path = './dummy/NoisyLR/'
    mpi_path = './dummy/ModelPI/'

    n_eval = len(os.listdir(noisylr_path))

    evaluator = ModelEvaluator(None, metrics, gt_path, noisylr_path, mpi_path, n_eval)
    evaluator.evalulate()
    evaluator.display()
