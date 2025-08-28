# DLAI_project

# VQ-VAE a Posteriori with Geodesic Quantization

This repository contains the code, final models, and figures for the "Deep Learning and Applied AI" course. The project explores an innovative method for a posteriori vector quantization in a Variational Autoencoder's (VAE) latent space using geodesic distances, and compares it against a standard end-to-end VQ-VAE pipeline.

## Key Concepts Explored

* **Continuous VAE with a Latent Grid (`GridVAE`)**: Implementation of a standard VAE that learns a continuous, spatially organized latent space.
* **A Posteriori Geodesic Clustering**: A multi-stage quantization pipeline that includes:
    * Sampling of "landmark" vectors from the continuous latent space.
    * Construction of a k-NN graph to approximate the data manifold.
    * Calculation of a geodesic distance matrix using all-pairs-shortest-path.
    * `KMedoids` clustering with the precomputed geodesic metric to create a geometrically-aware codebook.
* **Standard VQ-VAE Baseline**: Implementation of an end-to-end VQ-VAE, including an analysis of the "codebook collapse" problem.
* **Autoregressive Prior**: Training of a decoder-only `Transformer` to learn the distribution of discrete codes for both pipelines and generate new samples.
* **Comparative Analysis**: A comprehensive comparison of the two pipelines on reconstruction quality (MSE), codebook utilization, sample fidelity (visual), and perplexity.

## File Structure

```
.
└── DLAI_project/
    ├── DLAI_Project.ipynb      <-- The main notebook.
    │
    ├── grid_vae.pt             <-- Trained weights for the GridVAE (Method A).
    ├── transformer_A.pt        <-- Trained weights for the Transformer of Method A.
    ├── vq_model_stable.pt      <-- Trained weights for the stable VQ-VAE (Method B).
    ├── transformer_B.pt        <-- Trained weights for the Transformer of Method B.
    │
    ├── clustering_comparison.jpg <-- Figure: Geodesic vs. Euclidean clustering.
    ├── codebook_utilization.png  <-- Figure: Codebook utilization comparison.
    ├── reconstruction.png        <-- Figure: Reconstruction quality comparison.
    └── sample_fidelity.png       <-- Figure: Generated sample fidelity comparison.
```

## How to Run

The project is contained entirely within the `DLAI_Project.ipynb` notebook and is designed to be run in Google Colab.

1.  **Run the first cell (Environment Setup)**. This cell will install the specific library versions required and then force an automatic kernel restart. **This is expected behavior.**
2.  **Wait for the kernel to restart**. You will see the RAM/Disk indicators in the top-right corner reconnect.
3.  Once reconnected, **run all subsequent cells in order**, starting from the second cell . The notebook will then execute the full training and analysis pipeline for both methods.

## Summary of Results

The experimental analysis revealed a fundamental trade-off between the two approaches.
* **Our innovative method (Method A)**, using a posteriori geodesic quantization, successfully produced a healthy and **100% utilized codebook**. This led to the generation of recognizable, albeit imperfect, digit-like samples.
* **The standard VQ-VAE baseline (Method B)**, while achieving slightly better reconstruction quality, suffered from a severe and persistent **codebook collapse**, utilizing only a small fraction of its available codes. This critically impaired its generative capabilities, resulting in low-fidelity, abstract samples.

The project concludes that the a posteriori geodesic approach, while more computationally complex, provides a more stable and robust foundation for generative modeling by avoiding the common pitfalls of end-to-end discrete representation learning.
