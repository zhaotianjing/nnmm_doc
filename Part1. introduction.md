# Part1. introduction

## 1. Overview
The Mixed Effect Neural Networks (NN-MM) extend linear mixed model ("MM") to multilayer neural networks ("NN") by adding one middle layer between genotype layer and phenotypes layer. Nodes in the middle layer represent intermediate traits, e.g., the known intermediate omics features such as gene expression levels can be incorporated in the middle layer. These three sequential layers form a unified network.


![](https://github.com/zhaotianjing/figures/blob/main/omics_example.png)


## 2. Extend linear mixed model ("MM") to multilayer neural networks ("NN")?

Multiple independent single-trait mixed models are used to model the relationships between input layer (genotypes) and middle layer (intermediate traits). Activation functions in the neural network are used to approximate the linear/nonlinear relationships between middle layer (intermediate traits) and output layer (phenotypes). Missing values in the middle layer (intermediate traits) are sampled by Hamiltonian Monte Carlo based on the upstream genotype layer and downstream phenotype layer.

Details can be found in our publications:

> * Tianjing Zhao, Jian Zeng, and Hao Cheng. Extend mixed models to multilayer neural networks for genomic prediction including intermediate omics data, GENETICS, 2022; [https://doi.org/10.1093/genetics/iyac034](https://doi.org/10.1093/genetics/iyac034). 
> * Tianjing Zhao, Rohan Fernando, and Hao Cheng. Interpretable artificial neural networks incorporating Bayesian alphabet models for genome-wide prediction and association studies, G3 Genes|Genomes|Genetics, 2021;  [https://doi.org/10.1093/g3journal/jkab228](https://doi.org/10.1093/g3journal/jkab228)

