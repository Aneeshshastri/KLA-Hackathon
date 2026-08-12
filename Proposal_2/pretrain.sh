conda run -n ml_env python train_expert.py --expert-type upsample --data-dir data/upsample_only --epochs 40
conda run -n ml_env python train_expert.py --expert-type deblur --data-dir data/blur_only --epochs 40
conda run -n ml_env python train_expert.py --expert-type gaussian --data-dir data/gaussian_only --epochs 40
conda run -n ml_env python train_expert.py --expert-type speckle --data-dir data/speckle_only --epochs 40
