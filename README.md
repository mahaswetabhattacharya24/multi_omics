#Multi Omics

This autoencoder script tries to find an underlying manifold for scRNAseq data obtained through generative model - a deep autoencoder.
The script does the following:
  i. Loads scRNAseq data
  ii. Identifies the highly variable genes
  iii. Performs data preprocessing
  iv. The model simulataneously does two things: a. Accurate reconstruction from the latent space using the autoencoder, b. Classification from the latent space
  v. Loss functions used are RMSE and binary cross entropy
