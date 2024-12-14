# Fuzzy Consensus Clustering Implementation

This repository contains a Python implementation of the Fuzzy Consensus Clustering (FCC) algorithm as described in the paper "Fuzzy Consensus Clustering With Applications on Big Data" by Wu et al. The implementation provides a framework for combining multiple clustering results into a consensus clustering using fuzzy logic principles.

## Overview

The Fuzzy Consensus Clustering algorithm combines multiple clustering results to create a more robust and reliable clustering solution. The implementation offers two ways to use the algorithm:

1. **Direct Consensus Formation** (`FCC_direct`): Use this if you already have fuzzy membership matrices from your own clustering algorithms or methods
2. **Full Pipeline** (`FCC`): Use this to perform the complete clustering process including GMM and FCM base clusterings

This flexibility makes the implementation suitable for:

- Working with pre-existing clustering results
- Analyzing complex datasets where single clustering algorithms might produce inconsistent results
- Combining multiple clustering perspectives to achieve more stable results
- Working with high-dimensional data through optional dimensionality reduction

## Requirements

The implementation requires the following Python packages:

- NumPy
- scikit-learn (for PCA and Gaussian Mixture Models) - only needed if using the full FCC pipeline
- FCM (Fuzzy C-Means implementation) - only needed if using the full FCC pipeline

## Usage

The implementation provides two main functions:

### 1. FCC_direct (Direct Consensus Function)

If you already have your own fuzzy membership matrices from any clustering method, you can use this function directly:

```python
FCC_direct(clusterings,
          final_cluster_no,
          output_file,
          fuzzy_factor=2,
          clustering_weights=None,
          tol=1e-4,
          max_iter=100)
```

Example usage with existing clustering results:

```python
import numpy as np

# Your pre-computed fuzzy membership matrices
clustering1 = np.array([[0.8, 0.2], [0.6, 0.4], ...])  # From method 1
clustering2 = np.array([[0.7, 0.3], [0.5, 0.5], ...])  # From method 2
clustering3 = np.array([[0.9, 0.1], [0.4, 0.6], ...])  # From method 3

# Combine your clustering results
clusterings = [clustering1, clustering2, clustering3]

# Run consensus clustering directly
FCC_direct(clusterings=clusterings,
          final_cluster_no=2,
          output_file='consensus_result.npy')
```

### 2. FCC (Full Pipeline Function)

This function handles the complete clustering pipeline including generating base clusterings:

```python
FCC(input_file,
    FCM_cluster_numbers,
    GMM_cluster_numbers,
    final_cluster_no,
    output_file,
    FCM_fuzzy_factor=2,
    FCC_fuzzy_factor=2,
    clustering_weights=None,
    PCA_n_components=None,
    tol=1e-4,
    max_iter=100)
```

Example usage:

```python
import numpy as np

# Prepare your data
data = np.array([[...], [...], ...])  # Your input data

# Define clustering parameters
fcm_clusters = [2, 3, 4]  # Try FCM with 2, 3, and 4 clusters
gmm_clusters = [2, 3, 4]  # Try GMM with 2, 3, and 4 clusters
final_clusters = 3        # Desired number of consensus clusters

# Run the algorithm
FCC(data,
    FCM_cluster_numbers=fcm_clusters,
    GMM_cluster_numbers=gmm_clusters,
    final_cluster_no=final_clusters,
    output_file='consensus_result.npy',
    PCA_n_components=50)  # Optional dimensionality reduction
```

## Algorithm Details

The implementation follows these steps:

1. **Input Processing**

   - For `FCC_direct`: Uses pre-computed fuzzy membership matrices directly
   - For `FCC`: Performs data preprocessing and base clustering generation

2. **Consensus Formation**
   - Initializes a random consensus matrix
   - Iteratively updates cluster centroids and membership values
   - Converges to a final consensus clustering

## Output

The algorithm saves the final consensus matrix as a NumPy array (.npy file) at the specified output path. The matrix contains membership values for each data point across the final clusters.

## Parameters

Key parameters that can be tuned:

- `clusterings`: List of fuzzy membership matrices (for FCC_direct)
- `final_cluster_no`: Desired number of clusters in the consensus
- `fuzzy_factor`: Fuzziness parameter for consensus clustering (default: 2)
- `clustering_weights`: Optional weights for each input clustering
- `tol`: Convergence tolerance (default: 1e-4)
- `max_iter`: Maximum iterations (default: 100)

Additional parameters for full FCC pipeline:

- `FCM_cluster_numbers`: List of cluster numbers for FCM
- `GMM_cluster_numbers`: List of cluster numbers for GMM
- `FCM_fuzzy_factor`: Fuzziness parameter for FCM (default: 2)
- `PCA_n_components`: Number of components for dimensionality reduction (optional)

## Citation

If you use this implementation in your research, please cite the original paper:

```
Wu, J., et al. "Fuzzy Consensus Clustering With Applications on Big Data"
```

## License

This project is licensed under the [MIT License](./LICENSE).
