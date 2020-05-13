<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Spiking VAE

## Table of content
- [Description](#description)
- [Author](#author)
- [Features](#features)
- [Results](#results)
- [Usage](#usage)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Description 

Thesis project for MSc. Artificial Intelligence at Vrije Universiteit Amsterdam. My aim is to implement a variational autoencoder with spiking neurons in in [PyTorch](https://github.com/pytorch/pytorch).

## Author

[Justus F. Hübotter](https://www.huebotter.net)

## Features

* [x] CPU/GPU support
* [x] TensorBoard real-time monitoring
* [x] MNIST dataset
* [x] Fashion-MNIST dataset
* [ ] CelebA dataset
* [ ] Bouncing balls dataset
* [x] Fully connected classifier network
* [x] Convolutional classifier network 
* [ ] Convolutional autoencoder
* [ ] Convolutional variational autoencoder
* [ ] Convolutional spiking autoencoder
* [ ] Convolutional spiking variational autoencoder
* [ ] Model weights automatically saved at checkpoints
* [ ] Pretrained weights available
* [ ] Fully commented and documented
* [ ] ...


## Results

Tests are done on Ubuntu 18.04 with 16 GB RAM, Ryzen 5 3600, nVidia RTX 2070.

## Usage

### Setup

>Requires Python 3.7

The following lines will clone the repository and install all the required dependencies.

```bash
$ https://github.com/jhuebotter/SpikingVAE.git
$ cd SpikingVAE
$ pip install -r requirements.txt
```

### Datasets

In order to download datasets used in the paper experiments use
```bash
$ python setup.py
```

with options `mnist` and `fashion`. For example, if case you want to replicate *all* experiments in this thesis, you must run the following line:

```bash
$ python setup.py mnist fashion
```

It will download and store the datasets locally in the **data** folder. 

### Pretrained Models


### Train Models 

```bash
$ cd src
$ python [model] [args] 
```

For example

```bash
$ python train_cnn.py --dataset mnist --epochs 10 --report-interval 1 --lr 0.001 
```

To visualize training results in TensorBoard, we can use the following command from a new terminal within **src** folder. 

```bash
$ tensorboard --logdir='results/logs'
```


## References

Related Papers:
- [Variational Sparse Coding](https://openreview.net/pdf?id=SkeJ6iR9Km)
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
- [Large-Scale Feature Learning With Spike-and-Slab Sparse Coding](https://arxiv.org/pdf/1206.6407.pdf)
- [Stick-Breaking Variational Autoencoders](https://arxiv.org/pdf/1605.06197.pdf)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
- [Disentangling by Factorising](https://arxiv.org/pdf/1802.05983.pdf)
- [Neural Discrete Representation Learning](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)
- [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)

## Acknowledgements 

## License

[MIT License](https://github.com/jhuebotter/SpikingVAE/blob/master/LICENSE)

