import numpy as np
from fcmeans import FCM
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def FCC_direct(clusterings, final_cluster_no, output_file, fuzzy_factor=2, clustering_weights=None, tol=1e-4, max_iter=100):
    """
    Performs Fuzzy Consensus Clustering (FCC) on pre-computed clustering results.

    This function implements a consensus clustering algorithm that combines multiple fuzzy clustering
    results into a single consensus clustering. It uses an iterative approach to optimize the consensus
    matrix until convergence or maximum iterations are reached.

    Parameters
    ----------
    clusterings : list of numpy.ndarray
        List of membership matrices from different clustering algorithms.
        Each matrix should have shape (n_samples, n_clusters_i) where n_clusters_i
        can vary between clustering results.
    final_cluster_no : int
        The desired number of clusters in the consensus result.
    output_file : str
        Path where the final consensus matrix will be saved as a .npy file.
    fuzzy_factor : float, optional (default=2)
        The fuzziness coefficient. Higher values lead to softer clustering.
        Must be greater than 1.
    clustering_weights : numpy.ndarray, optional (default=None)
        Weights for each input clustering. If None, equal weights are used.
    tol : float, optional (default=1e-4)
        Tolerance for convergence. The algorithm stops when the change in
        centroids is less than this value.
    max_iter : int, optional (default=100)
        Maximum number of iterations for the consensus algorithm.

    Returns
    -------
    None
        The consensus matrix is saved to the specified output_file.

    Notes
    -----
    The function uses helper methods for:
    - Initializing the consensus matrix
    - Calculating cluster centroids
    - Updating consensus matrix
    - Computing distances between samples and centroids
    """
    def initialize_consensus_matrix(n, final_cluster_no):
        """
        Initialize the consensus matrix with random values normalized by row.

        Parameters
        ----------
        n : int
            Number of samples
        final_cluster_no : int
            Number of clusters in consensus

        Returns
        -------
        numpy.ndarray
            Initialized consensus matrix of shape (n, final_cluster_no)
        """
        matrix = np.random.rand(n, final_cluster_no)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        return matrix

    def get_mu_l_dot_i(clusterings, clustering_idx, row_idx):
        return clusterings[clustering_idx][row_idx]

    def calculate_vki(consensus, clusterings, clustering_idx, fuzzy_factor, k):
        vki = 0
        n = consensus.shape[0]

        for l in range(n):
            mu_l_k = consensus[l][k]
            mu_dot_k = consensus[:, k]
            mu_l_dot_i = get_mu_l_dot_i(clusterings, clustering_idx, l)
            vki += (mu_l_k ** fuzzy_factor) / np.linalg.norm(mu_dot_k ** fuzzy_factor, ord=1) * mu_l_dot_i

        return vki

    def calculate_vk(consensus, clusterings, fuzzy_factor, k):
        vk = []

        for i in range(len(clusterings)):
            vki = calculate_vki(consensus, clusterings, i, fuzzy_factor, k)
            vk.append(vki)

        return vk

    def update_centroids(consensus, clusterings, fuzzy_factor, final_cluster_no):
        new_centroids = []

        for k in range(final_cluster_no):
            vk = calculate_vk(consensus, clusterings, fuzzy_factor, k)
            new_centroids.append(vk)

        return new_centroids

    def calculate_dist(centroids, clusterings, clustering_weights, l, k):
        dist = 0

        for i in range(len(clusterings)):
            mu_l_dot_i = get_mu_l_dot_i(clusterings, i, l)
            vki = centroids[k][i]
            l2_norm = np.linalg.norm(mu_l_dot_i - vki) ** 2
            dist += clustering_weights[i] * l2_norm

        return dist

    def update_consensus(consensus, centroids, clusterings, clustering_weights, fuzzy_factor, epsilon=1e-16):
        rows, cols = consensus.shape
        dist_matrix = np.zeros((rows, cols))

        for l in range(rows):
            for k in range(cols):
                dist = calculate_dist(centroids, clusterings, clustering_weights, l, k)
                dist_matrix[l][k] = dist

        consensus_copy = consensus.copy()

        for l in range(rows):
            for k in range(cols):
                consensus_copy[l][k] = (dist_matrix[l][k] + epsilon) ** (-1 / (fuzzy_factor - 1)) / np.sum((dist_matrix[l] + epsilon) ** (-1 / (fuzzy_factor - 1)))

        return consensus_copy

    def calculate_centroid_difference(old_centroids, new_centroids):
        total_diff = 0

        for k in range(len(old_centroids)):
            old_cluster = old_centroids[k]
            new_cluster = new_centroids[k]

            for i in range(len(old_cluster)):
                old_center = old_cluster[i]
                new_center = new_cluster[i]

                diff = np.linalg.norm(np.array(old_center) - np.array(new_center))
                total_diff += diff

        return total_diff

    n = clusterings[0].shape[0]

    if clustering_weights is None:
        no_of_bps = len(clusterings)
        clustering_weights = np.full(no_of_bps, 1 / no_of_bps)

    # Initialize consensus and centroids
    consensus = initialize_consensus_matrix(n, final_cluster_no)
    centroids = update_centroids(consensus, clusterings, fuzzy_factor, final_cluster_no)

    for _ in range(max_iter):
        consensus = update_consensus(consensus, centroids, clusterings, clustering_weights, fuzzy_factor)
        new_centroids = update_centroids(consensus, clusterings, fuzzy_factor, final_cluster_no)

        if calculate_centroid_difference(centroids, new_centroids) < tol:
            np.save(output_file, consensus)
            return

        centroids = new_centroids

    np.save(output_file, consensus)


def FCC(input_file, FCM_cluster_numbers, GMM_cluster_numbers, final_cluster_no, output_file, FCM_fuzzy_factor=2, FCC_fuzzy_factor=2, clustering_weights=None, PCA_n_components=None, tol=1e-4, max_iter=100):
    """
    Performs Fuzzy Consensus Clustering (FCC) on input data using both GMM and FCM as base clusterings.

    This function first preprocesses the input data, optionally reduces dimensionality using PCA,
    then applies multiple Gaussian Mixture Models (GMM) and Fuzzy C-Means (FCM) clusterings
    before combining them using the FCC_direct method.

    Parameters
    ----------
    input_file : numpy.ndarray
        Input data matrix of shape (n_samples, n_features).
    FCM_cluster_numbers : list of int
        List of cluster numbers to use for FCM clustering.
    GMM_cluster_numbers : list of int
        List of cluster numbers to use for GMM clustering.
    final_cluster_no : int
        The desired number of clusters in the consensus result.
    output_file : str
        Path where the final consensus matrix will be saved.
    FCM_fuzzy_factor : float, optional (default=2)
        Fuzziness parameter for FCM clustering.
    FCC_fuzzy_factor : float, optional (default=2)
        Fuzziness parameter for the consensus clustering.
    clustering_weights : numpy.ndarray, optional (default=None)
        Weights for each base clustering. If None, equal weights are used.
    PCA_n_components : int, optional (default=None)
        Number of components for PCA dimensionality reduction.
        If None, no dimensionality reduction is performed.
    tol : float, optional (default=1e-4)
        Tolerance for convergence in FCC_direct.
    max_iter : int, optional (default=100)
        Maximum number of iterations for FCC_direct.

    Returns
    -------
    None
        The consensus matrix is saved to the specified output_file.

    Notes
    -----
    The function performs the following steps:
    1. Normalizes input data to probability distributions
    2. Optionally performs PCA dimensionality reduction
    3. Applies multiple GMM clusterings
    4. Applies multiple FCM clusterings
    5. Combines all clustering results using FCC_direct
    """
    # Convert raw counts to probability distribution
    row_sums = input_file.sum(axis=1, keepdims=True)
    input_file = input_file / row_sums

    # Perform dimensionality reduction
    if PCA_n_components is not None:
        pca = PCA(n_components=PCA_n_components)
        pca.fit(input_file)
        input_file = pca.transform(input_file)

    clusterings = []

    # Perform GMM
    for i in range(len(GMM_cluster_numbers)):
        gmm = GaussianMixture(n_components=GMM_cluster_numbers[i])

        gmm.fit(input_file)

        GMM_membership_matrix = gmm.predict_proba(input_file)

        clusterings.append(GMM_membership_matrix)

    # Perform FCM
    for i in range(len(FCM_cluster_numbers)):
        fcm = FCM(n_clusters=FCM_cluster_numbers[i], m=FCM_fuzzy_factor)

        fcm.fit(input_file)

        FCM_membership_matrix = fcm.soft_predict(input_file)

        clusterings.append(FCM_membership_matrix)

    return FCC_direct(clusterings, final_cluster_no, output_file, FCC_fuzzy_factor, clustering_weights, tol, max_iter)
