

<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Spiking VAE

## Table of content
- [Description](#description)
- [Author](#author)
- [Example](#example)
- [Features](#features)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Description 

This is my thesis project for the MSc. Artificial Intelligence program at Vrije Universiteit Amsterdam. I implemented an autoencoder network with spiking neurons in [PyTorch](https://github.com/pytorch/pytorch). 

## Author

Justus F. Hübotter

## Example

<img src="https://github.com/jhuebotter/SpikingVAE/blob/master/methods.png" alt="Image reconstruction example" title="Image reconstruction example" width="500"/>

The regularized spiking autoencoder model encodes to and decodes from a spiking latent representation.

<img src="https://github.com/jhuebotter/SpikingVAE/blob/master/SAE_dense_reconstruction.gif" alt="Image reconstruction example" title="Image reconstruction example" width="900"/>

Spiking model performs the iamge reconstruction task well under the influence of noisey inputs.

<img src="https://github.com/jhuebotter/SpikingVAE/blob/master/SAE_dense_reconstruction_noise.gif" alt="Image reconstruction example" title="Image reconstruction example" width="900"/>


## Features

* [x] CPU/GPU support

* [x] TensorBoard real-time monitoring
* [x] Weights and Biases logging
* [x] Custom loss functions
* [x] Custom metrics 
* [x] Best and last model weights automatically saved
* [ ] Pretrained weights available
* [x] Reconstruction & representation plotting
* [ ] Dataset preprocessing options
* [x] Fully commented and documented

* [x] MNIST dataset
* [x] Fashion-MNIST dataset
* [ ] CelebA dataset
* [ ] Bouncing balls dataset
* [ ] Moving MNIST dataset
* [x] Image-to-spike encoding (rate and time code)

* [x] Fully parameterized model architectuce
* [x] Fully connected classifier
* [x] Convolutional classifier 
* [x] Fully connected spiking classifier
* [x] Spiking convolutional classifier 
* [x] Fully connected autoencoder
* [x] Convolutional autoencoder
* [x] Fully connected variational autoencoder
* [x] Convolutional variational autoencoder
* [x] Fully connected spiking autoencoder
* [x] Convolutional spiking autoencoder


## Results

Development and Testing are done on Ubuntu 18.04 with 16 GB RAM, Ryzen 5 3600, nVidia RTX 2070.

## Usage

### Setup

>Requires Python 3.7

The following lines will clone the repository and install all the required dependencies.

```bash
$ https://github.com/jhuebotter/SpikingVAE.git
$ cd SpikingVAE
$ pip install -r requirements.txt
```

This project uses Weights and Biases for logging. In order to use this package, having an account with their platform is mandatory. Before running the scrips, you to login from you local machine by running

```bash
$ wandb login
```

For more information please see the [official documentation](https://docs.wandb.com/quickstart).


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

Are not yet available here.

### Train Models 

```bash
$ cd src
$ python [model] [args] 
```

For example

```bash
$ python train_cnn.py --dataset mnist --epochs 10 --report-interval 1 --lr 0.001 
```

## License

[MIT License](https://github.com/jhuebotter/SpikingVAE/blob/master/LICENSE)

