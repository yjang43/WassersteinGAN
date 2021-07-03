# WGAN

This repository and to see a hands-on comparison between normal DCGAN and WGAN.

TODO: Add WGAN-GP method and etc.

WGAN allows stable training and prevents mode collapse. Check [WGAN paper](https://arxiv.org/pdf/1701.07875.pdf).

---

### Install

Clone the repository and install required packages.

```bash
git clone https://github.com/yjang43/WassersteinGAN.git
cd WassersteinGAN/
pip install -r requirements.txt
```

### Usage

To train, you need wandb to track the result

```bash
python train.py
# wandb prompt
# enter '3' if you do not have wandb account
```

Run examples:

```bash
python train.py                               # default uses WGAN, CIFAR-10 
python train.py --dataset_name mnist --nc 1   # if MNIST, make sure to set nc to 1
python train.py --model_name dcgan            # change train model to DCGAN
python train.py --batch_size 32               # set batch size for GPU memory
```


### Result 
#### Losses and Inception Score on CIFAR-10

__DCGAN__

<p>
    <img src="imgs/dcgan_g_loss.png" alt="dcgan_g_loss" width="400">
    <img src="imgs/dcgan_d_loss.png" alt="dcgan_d_loss" width="400">
    <img src="imgs/dcgan_is.png" alt="dcgan_is" width="400">
</p>

__WGAN__

<p>
    <img src="imgs/wgan_g_loss.png" alt="wgan_g_loss" width="400">
    <img src="imgs/wgan_d_loss.png" alt="wgan_d_loss" width="400">
    <img src="imgs/wgan_is.png" alt="wgan_is" width="400">
</p>

_inception score seems to be a bit off..._


#### Generated Images on CIFAR-10
DCGAN on the left and WGAN on the right

__500 itr__

![dcgan500](imgs/dcgan500.png)
![wgan500](imgs/wgan500.png)

__2000 itr__

![dcgan2000](imgs/dcgan2000.png)
![wgan2000](imgs/wgan2000.png)

__5000 itr__

![dcgan5000](imgs/dcgan5000.png)
![wgan5000](imgs/wgan5000.png)

__20000 itr__<br>
Although not promising with small number of iteration, DCGAN image quality gets better

![dcgan20000](imgs/dcgan20000.png)
![wgan20000](imgs/wgan20000.png)



#### Generated Images on MNIST
DCGAN on the left and WGAN on the right

__500 itr__

![dcgan500mnist](imgs/dcgan500mnist.png)
![wgan500mnist](imgs/wgan500mnist.png)

__2000 itr__

![dcgan2000mnist](imgs/dcgan2000mnist.png)
![wgan2000mnist](imgs/wgan2000mnist.png)

__5000 itr__

![dcgan5000mnist](imgs/dcgan5000mnist.png)
![wgan5000mnist](imgs/wgan5000mnist.png)


### Reference
__GAN__

https://github.com/Zeleni9/pytorch-wgan


__Inception Score v3__

https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py



