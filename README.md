# Gaussian Mixture Variational Autoencoders with Separation Regularization for Enhanced Clustering Performance

This project presents a comparative study between **Gaussian Mixture Variational Autoencoders (GMVAE)** and **Beta-Variational Autoencoders (Beta-VAE)** for unsupervised clustering and latent representation learning. The work investigates how disentanglement and mixture priors influence clustering quality, stability, and representation robustness.

## Project Overview

Variational Autoencoders (VAEs) are widely used for unsupervised representation learning, but standard VAEs often struggle with entangled latent spaces and weak clustering behavior. This project explores two advanced VAE variants:

* **GMVAE** → Introduces Gaussian mixture priors for cluster-aware latent representations
* **Beta-VAE** → Enforces disentangled latent factors by scaling the KL divergence

The project evaluates both approaches on clustering tasks using multiple unsupervised learning metrics.

## Research Objective

The primary research question addressed in this work:

> Which approach performs better for clustering and representation learning — GMVAE or Beta-VAE?

The study focuses on:

* Clustering quality
* Latent space disentanglement
* Stability across runs
* Representation interpretability

## Model Architectures

### GMVAE Components

The proposed GMVAE architecture includes:

* Encoder Network
* Gaussian Mixture Prior
* Decoder Network
* Separation Regularization Module

### Beta-VAE

Beta-VAE modifies the standard VAE objective by introducing a weighted KL-divergence term to encourage disentangled latent representations.

## Mathematical Formulation

### Modified ELBO Objective

\mathcal{L}(\theta,\phi;x)=\mathbb{E}*{q*\phi(z|x)}[\log p_\theta(x|z)]-\beta D_{KL}(q_\phi(z|x)|p(z))+\gamma \mathcal{L}_{sep}

### Gaussian Mixture Prior

p(z)=\sum_{k=1}^{K}\pi_k\mathcal{N}(z|\mu_k,\sigma_k^2)

### Separation Regularizer

\mathcal{L}*{sep}=-\mathbb{E}*{i\neq j}\left[\frac{1}{|\mu_i-\mu_j|^2+\epsilon}\right]

The separation regularizer was introduced to reduce cluster collapse by encouraging cluster centers to remain distinct in latent space.

## Experimental Setup

### Dataset

A benchmark dataset for unsupervised clustering was used with preprocessing techniques including:

* Normalization
* Dimensionality reduction (when necessary)

### Training Configuration

| Parameter      | Value         |
| -------------- | ------------- |
| Optimizer      | Adam          |
| Learning Rate  | 0.001         |
| Batch Size     | 128           |
| Maximum Epochs | 100           |
| Early Stopping | Patience = 10 |

### Frameworks & Environment

* Python 3.10
* PyTorch
* NVIDIA GPU
* 32GB RAM

## Evaluation Metrics

The models were evaluated using:

* Accuracy (ACC)
* Normalized Mutual Information (NMI)
* Adjusted Rand Index (ARI)
* Silhouette Score
* Stability Across Runs

## Results

| Model    | ACC    | NMI    | ARI    | Silhouette |
| -------- | ------ | ------ | ------ | ---------- |
| GMVAE    | 0.3600 | 0.4015 | 0.2145 | 0.0902     |
| Beta-VAE | 0.5960 | 0.5696 | 0.4159 | 0.1272     |

### Key Findings

* Beta-VAE consistently outperformed GMVAE across all evaluation metrics
* Beta-VAE demonstrated higher training stability
* GMVAE suffered from:

  * Cluster collapse
  * Sensitivity to initialization
  * Lower disentanglement quality

The results suggest that disentangled latent representations contribute more effectively to clustering performance than explicitly modeled mixture priors.

## Discussion

Although GMVAE theoretically aligns latent space with cluster structures, practical limitations reduce its effectiveness. Beta-VAE’s stronger disentanglement capability produced more robust and interpretable latent representations, leading to improved clustering behavior.

This indicates that:

* Disentanglement may be more critical than mixture priors
* Stable latent factorization improves unsupervised representation learning

## Future Work

Potential future improvements include:

* Hybrid GMVAE + Beta-VAE architectures
* Advanced disentanglement regularization
* Improved anti-collapse mechanisms
* Experiments on larger benchmark datasets

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn

## References

* Auto-Encoding Variational Bayes
* Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
* beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework

## Author

**Arif Bin Islam**
Department of Computer Science and Engineering
BRAC University

## Course Information

* Course: CSE425
* Submission Date: September 2025

## Source Code

Add your GitHub repository link here:

```bash
https://github.com/Arif-gits/CSE425-Neural-Network-Design-and-implement-a-non-deterministic-unsupervised-neural-network-model
```
