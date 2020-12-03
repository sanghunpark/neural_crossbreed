# Neural Crossbreed: Neural Based Image Metamorphosis
[[Paper]](https://arxiv.org/abs/2009.00905)

<p align="left">
<img src="docs/teaser.png">
</p>

__Abstract__ We propose Neural Crossbreed, a feed-forward neural network that can learn a semantic change of input images in a latent space to create the morphing effect. Because the network learns a semantic change, a sequence of meaningful intermediate images can be generated without requiring the user to specify explicit correspondences. In addition, the semantic change learning makes it possible to perform the morphing between the images that contain objects with significantly different poses or camera views. Furthermore, just as in conventional morphing techniques, our morphing network can handle shape and appearance transitions separately by disentangling the content and the style transfer for rich usability. We prepare a training dataset for morphing using a pre-trained BigGAN, which generates an intermediate image by interpolating two latent vectors at an intended morphing value. This is the first attempt to address image morphing using a pre-trained generative model in order to learn semantic transformation. The experiments show that Neural Crossbreed produces high quality morphed images, overcoming various limitations associated with conventional approaches. In addition, Neural Crossbreed can be further extended for diverse applications such as multi-image morphing, appearance transfer, and video frame interpolation.


<!-- The code will be updated soon. -->
# Requirements
Python 3.7.4

PyTorch 1.2.0

TensorBoard

PyYAML 5.1.2

tqdm 4.41.1

PyTorch pretrained BigGAN 0.1.1

NLTK 3.4.5

scikit-image 0.16.2

# Installlation 

Clone this repository.
```bash
git clone https://github.com/sanghunpark/neural_crossbreed.git
cd neural_crossbreed/
```
>## For [Conda](https://www.anaconda.com/) users:
>```bash
>conda create -n neural_crossbreed python=3.7.4
>conda activate neural_crossbreed
>```

Install PyTorch 1.2.0+ and torchvision from [PyTorch](http://pytorch.org.)

Install other dependencies from the requirement file.

```bash
pip install -r requirements.txt 
```


# Testing

Download the checkpoint file of the pre-trained dog model from the [link](https://drive.google.com/drive/folders/1IhxQ-fus-maSEakuFy7PorP1dkWI1WyR?usp=sharing), save it in `./train_dir/nc_final/`.

To generate morphed images using the pre-trained dog model, run following command:
```bash
python test.py  --config=./config.yaml
                --ngpu=1
                --gpu_1st=0
                --input_a=./sample_images/dog5.png
                --input_b=./sample_images/dog3.png
                --niter=5
```
Generated images will be placed under `./test_outdir/out_***.png`.
<!-- Generated images will be stored in the `./test_outdir/`. -->

You can obtain the disentangled transition results by adding `--disentangled` and `--tau`. For example,
```bash
python test.py  --config=./config.yaml
                --ngpu=1
                --gpu_1st=0
                --input_a=./sample_images/dog5.png
                --input_b=./sample_images/dog3.png
                --niter=5
                --disentangled
                --tau=0.3
```

# Training
To train Neural Crossbreed from scratch, run the following command.

```bash
python train.py --config=./config.yaml
                --ngpu=[NUM_GPUs]
                --gpu_1st=[1ST_GPU_ID]
```
You can specify the GPU usage with the argument `--ngpu` and `--gpu_1st`. For instance, `--ngpu=2` and `--gpu_1st=3` will run the training process on GPU3 and GPU4. If you don't use these arguments, it will use all visible GPUs by running:
```bash
CUDA_AVAILABLE_DEVICES=0,1,... python train.py --config=./config.yaml
```

The training configuration such as path, model, and hyperparamters can be further customized with [`./config.yaml`](https://github.com/sanghunpark/neural_crossbreed/blob/master/config.yaml). Please see refer to [`./config.yaml`](https://github.com/sanghunpark/neural_crossbreed/blob/master/config.yaml). 

In particular, to resume training, replace the checkpoint path in the [`./config.yaml`](https://github.com/sanghunpark/neural_crossbreed/blob/master/config.yaml) file with the latest checkpoint. 
```yaml
# path
...
checkpoints: [MODEL_DIR]/[YOUR_CHECKPOINT].pt
...
```
If a checkpoint file exists in the path, training will resume at the point where a previous training run left off. Otherwise, the network will be trained from scratch.

## Citation
 If you find this work useful, please cite our paper:
 ```
@article {park2020neural,
    title = {Neural Crossbreed: Neural Based Image Metamorphosis},
    author = {Sanghun Park and Kwanggyoon Seo and Junyong Noh},
    journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2020)},
    volume = {39},
    number = {6},
    year = {2020}
}
```

## Acknowledgements
This repository contains pieces of code from the following repositories:

[PyTorch pretrained BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) and [FUNIT](https://github.com/NVlabs/FUNIT). 