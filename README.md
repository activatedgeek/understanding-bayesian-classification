# Understanding Bayesian Classification

[![](https://img.shields.io/badge/arXiv-2203.16481-red)](https://arxiv.org/abs/2203.16481)

This repository hosts the code to reproduce the results presented in the paper [On Uncertainty, Tempering, and Data Augmentation in Bayesian Classification](https://arxiv.org/abs/2203.16481) by [Sanyam Kapoor](https://sanyamkapoor.com), [Wesley J Maddox](https://wjmaddox.github.io), [Pavel Izmailov](https://izmailovpavel.github.io), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## Key Ideas

Aleatoric uncertainty captures the inherent randomness of the data, such as measurement
noise. In Bayesian regression, we often use a Gaussian observation model, where we control
the level of aleatoric uncertainty with a noise variance parameter. By contrast, for Bayesian
classification we use a categorical distribution with no mechanism to represent our beliefs
about aleatoric uncertainty. Our work shows that:

* Explicitly accounting for *aleatoric uncertainty* significantly improves the performance of Bayesian neural networks.

| <img src="https://i.imgur.com/2SQsGTH.png" alt="Aleatoric Conceptual" width="30%" > |
| --- |
| In classification problems, we do not have a direct way to specify our assumptions about aleatoric uncertainty. In particular, we might use the same Bayesian neural network model if we know the data contains label noise (scenario A) and if we know that there is no label noise (scenario B), leading to poor performance in at least one of these scenarios. |

* We can match or exceed the performance of posterior tempering by using a Dirichlet observation model, where we explicitly control the level of aleatoric uncertainty, without any need for tempering.

| ![Tiny-Imagenet](https://i.imgur.com/n8oV240.png) |
| --- |
| Accounting for the label noise via the noisy Dirichlet model or the tempered softmax likelihood significantly improves accuracy and test negative log likelihood accross the board, here shown for the Tiny Imagenet dataset. The optimal performance is achieved for different values of temperature in the tempered softmax likelihood and the noise parameter for the noisy Dirichlet likelihood. |

* The cold posterior effect is effectively eliminated by properly accounting for aleatoric uncertainty in the likelihood model.

| <img src="https://i.imgur.com/dlKRLDn.png" alt="Cold Posterior Effect" width="50%"> |
| --- |
| BMA test accuracy for the noisy Dirichlet model with noise parameter 1e−6 and the softmax likelihood as a function of posterior temperature on CIFAR-10. The noisy Dirichlet model shows no cold posterior effect. |

## Setup

All requirements are listed in [environment.yml](./environment.yml). Create a `conda` environment using:
```
conda env create -n <env_name>
```

Next, ensure Python modules under the `src` folder are importable as,
```
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

To use [bnn_priors](https://github.com/ratschlab/bnn_priors), see
respective [installation](https://github.com/ratschlab/bnn_priors#installation) instructions.

## Usage

The main script to run all SGMCMC experiments is [experiments/train_lik.py](./experiments/train_lik.py).

As an example, to run cyclical SGHMC with our proposed noisy Dirichlet likelihood on CIFAR-10 with label noise, run:
```shell
python experiments/train_lik.py --dataset=cifar10 \
                                --label_noise=0.2 \
                                --likelihood=dirichlet \
                                --noise=1e-2 \
                                --prior-scale=1 \
                                --sgld-epochs=1000 \
                                --sgld-lr=2e-7 \
                                --n-cycles=50 \
                                --n-samples=50
```

Each argument to the `main` method can be used as a command line argument due to [Fire](https://google.github.io/python-fire/guide/).
[Weights & Biases](https://docs.wandb.ai) is used for all logging.
Configurations for various Weights & Biases sweeps are also available under [configs](./configs).

## License

Apache 2.0
