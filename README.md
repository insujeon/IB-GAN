# The PyTorch implementation of IB-GAN model of AAAI 2021

This package contains a PyTorch implementation of IB-GAN presented in the submitted [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16967) in AAAI 2021.

You can reproduce the experiment on dSprite (Color-dSprite, 3DChairs, and CelebA) dataset with the this code.

Current implementation is based on python==1.4.0. Please refer environments.yml for the environment settings.

Please refer to the Technical appendix page for more detailed information of hypter parameter settings for each experiment.


## Contents

* Main code for dsprites (and cdsprite): "main.py"
* IB-GAN model for dsprites (and cdsprite): "./model/model.py"
* Disentanglement Evaluation codes for dsprites (and cdsprite): "evaluator.py", "checkout_scores.ipynb"

* Main code for 3d Chairs (and CelebA): "main2.py" 
* IB-GAN model for dsprites (and cdsprite): "./model/model2.py"


## Visdom for visualization

Since the defulat visidom option for main.py is True, you first want to run Visidom server berfore excuting the main program by typing
```
python -m visdom.server -p 8097
```
Then you can observe the visualization of the "convergence plot and generated samples" for each training iterations from 
```
localhost:8097
```


## Reproducing dSprite experiment

* dSprite dataset : "./data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

You can reproduce dSprite expreiment by typing:
```
python -W ignore main.py --seed 7 --z_dim 16 --r_dim 10 --batch_size 64 --optim rmsprop --dataset dsprites --viz True --viz_port 8097 --z_bias 0 --viz_name dsprites --beta 0.141 --alpha 1 --gamma 1 --G_lr 5e-5 --D_lr 1e-6 --max_iter 150000 --logiter 500 --ptriter 2500 --ckptiter 2500 --load_ckpt -1 --init_type normal --save_img True
```
Note, all the default parameter settings are optimally set up for the dSprite experiment (in the "main.py" file).
For more details on the parameter settings for other datasets, please refer to the Technical appendix.


* dSprite dataset for Kim's disentanglement score evaluation : Evauation file is currently not available. (will be update soon)
The evaulation process and code is same as cdsprite experiment.


## Reproducing Color-dSprite expreiemnt

* Color-dSprite dataset : Color dSprite Dataset is currently not available.

But you can create Colored-dSprites dataset by changing RGB channel of the original dsprites dataset.

Each channel of RGB takes 8 discrete values as : [0.00, 36.42, 72.85, 109.28, 145.71, 182.14, 218.57, 255.00] )

Then move Color-dSprites datset (eg. cdsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz) npz file to the folder (./data/dsprites-dataset/)

Run the code with following argument:
```
python -W ignore main.py --seed 7 --z_dim 16 --r_dim 10 --batch_size 64 --optim rmsprop --dataset cdsprites --viz True --viz_port 8097 --z_bias 0 --viz_name dsprites --beta 0.071 --alpha 1 --gamma 1 --G_lr 5e-5 --D_lr 1e-6 --max_iter 500000 --logiter 500 --ptriter 2500 --ckptiter 2500 --load_ckpt -1 --init_type normal --save_img True
```

* Color-dSprite dataset for Kim's disentanglement score evaluation : "./data/img4eval_cdsprites.7z". 

You first need to unzip "imgs4eval_cdsprites.7z" file using 7za.
Please locate all the unzip files in "/data/imgs4eval_cdsprites/*" folder. 

run the evaluation on Kim's disentanglment metric, type
```
python evaluator.py --dset_dir data/imgs4eval_cdsprites --logiter 5000 --lastiter 500000 --name main
```

After all the evaluations for each checkpoint is done, you can see the overall disentanglement scores with the "checkout_scores.ipynb" (jupyter notebook) file.
or you can just type
```
import os
import torch
torch.load('checkpoint/main/result.metric')
```
 to see the scores in the python console.
Then move Color-dSprites datset (eg. cdsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz) to ./data/dsprites-dataset/


## Reproducing CelebA experiment

* CelebA dataset : please download CelebA dataset and prepare 64x64 center cropped image files into the folder (./data/CelebA/cropped_64)

Then run the code with following argument:
```
python -W ignore main2.py --seed 0 --z_dim 64 --r_dim 15 --batch_size 64 --optim rmsprop --dataset celeba --viz_port 8097 --z_bias 0 --r_weight 0 --viz_name celeba --beta 0.35 --alpha 1 --gamma 1 --max_iter 1000000 --G_lr 5e-5 --D_lr 2e-6 --R_lr 5e-5 --ckpt_dir checkpoint --output_dir output --logiter 500 --ptriter 20000 --ckptiter 20000 --ngf 64 --ndf 64 --label_smoothing True --instance_noise_start 0.5 --instance_noise_end 0.01 --init_type orthogonal
```


## Reproducing 3dChairs experiment

* 3dChairs dataset : please download 3dChairs dataset and move image files into the folder (./data/3DChairs/images)
 
```
python -W ignore main2.py --seed 0 --z_dim 64 --r_dim 10 --batch_size 64 --optim rmsprop --dataset 3dchairs --viz_port 8097 --z_bias 0 --r_weight 0 --viz_name 3dchairs --beta 0.325 --alpha 1 --gamma 1 --max_iter 700000 --G_lr 5e-5 --D_lr 2e-6 --R_lr 5e-5 --ckpt_dir checkpoint --output_dir output --logiter 500 --ptriter 20000 --ckptiter 20000 --ngf 32 --ndf 32 --label_smoothing True --instance_noise_start 0.5 --instance_noise_end 0.01 --init_type orthogonal
```


## Citing IB-GAN

If you like this work and end up using IB-GAN for your reseach, please cite our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16967) with the bibtex code:

@inproceedings{jeon2021ib,
  title={IB-GAN: Disengangled Representation Learning with Information Bottleneck Generative Adversarial Networks},
  author={Jeon, Insu and Lee, Wonkwang and Pyeon, Myeongjang and Kim, Gunhee},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={7926--7934},
  year={2021}
}

The disclosure and use of the currently published code is limited to research purposes only.


