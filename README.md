# Bayesian data selection
This repo provides example code implementing Bayesian data selection with the "Stein volume criterion (SVC)", as introduced in the paper

> Bayesian data selection, Eli N. Weinstein and Jeffrey W. Miller, 2021

## Installation

Download this repo, create a new python 3 virtual environment (eg. using conda), and run

    pip install .

To test your installation, navigate to the `svc` subfolder and run

    pytest

## pPCA

To perform data selection on a probabilistic PCA model, using the fast linear approximation described in the paper, navigate to the `svc` subfolder and run

    python pPCA.py example_pPCA.cfg

A detailed description of the model's options and how to input your own data can be found in the config file `example_pPCA.cfg`.

## Glass

To perform data selection on the glass model of expression data described in the paper, using a variational approximation to the SVC and the LOORF estimator, navigate to the `svc` subfolder and run

    python RNAGlass.py example_RNAGlass.cfg

A detailed description of the model's options and how to input your own data can be found in the config file `example_RNAGlass.cfg`.
