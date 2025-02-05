import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import random
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from collections import defaultdict

def generate_bundle_matrix(bundles, whole_items):
    """
    Generates a B x N binary matrix from a list of bundles and whole_items list.

    Args:
        bundles (list of sets): List of bundles, where each bundle is a set of item IDs.
        whole_items (list): List of all unique item IDs.

    Returns:
        np.ndarray: Binary matrix of size B x N.
    """
    item_index = {item: i for i, item in enumerate(whole_items)}  # Map items to column indices
    num_bundles = len(bundles)
    num_items = len(whole_items)

    matrix = np.zeros((num_bundles, num_items), dtype=int)

    for b, bundle in enumerate(bundles):
        for item in bundle:
            matrix[b, item_index[item]] = 1
    
    return matrix


def k_means_clustering(matrix, k, max_iters=100, tolerance=1e-4, random_seed=42):
    """
    Performs k-means clustering on a binary matrix.

    Args:
        matrix (np.ndarray): B x N binary matrix, where B is the number of bundles and N is the number of items.
        k (int): Number of clusters to form.
        max_iters (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (cluster_assignments, cluster_centers)
            cluster_assignments (np.ndarray): Array of cluster labels for each row in the matrix.
            cluster_centers (np.ndarray): Binary matrix of size k x N representing the cluster centers.
    """
    np.random.seed(random_seed)
    num_bundles, num_items = matrix.shape

    # Initialize k random cluster centers
    cluster_centers = matrix[np.random.choice(num_bundles, k, replace=False)]

    for iteration in range(max_iters):
        # Compute distances between each bundle and cluster centers
        distances = pairwise_distances(matrix, cluster_centers, metric="hamming")
        
        # Assign each bundle to the nearest cluster
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Compute new cluster centers
        new_cluster_centers = np.zeros_like(cluster_centers, dtype=float)
        for cluster in range(k):
            cluster_points = matrix[cluster_assignments == cluster]
            if len(cluster_points) > 0:
                # Average over cluster points and threshold to maintain binary values
                new_cluster_centers[cluster] = (cluster_points.mean(axis=0) > 0.5).astype(int)
            else:
                # If a cluster has no points, reinitialize it randomly
                new_cluster_centers[cluster] = matrix[np.random.choice(num_bundles)]

        # Check for convergence
        if np.allclose(cluster_centers, new_cluster_centers, atol=tolerance):
            break

        cluster_centers = new_cluster_centers

    return cluster_assignments, cluster_centers


import networkx as nx
import numpy as np

def min_k_cut(graph, k, seed=None):
    """
    Compute the minimum k-cut of an undirected graph.

    Args:
        graph (nx.Graph): The input graph.
        k (int): The number of clusters (cuts) to partition the graph into.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (cut_value, partitions), where:
            - cut_value (float): The total weight of the minimum k-cut.
            - partitions (list of sets): A list of k disjoint partitions of the nodes.
    """
    if k < 2 or k > graph.number_of_nodes():
        raise ValueError("k must be at least 2 and no greater than the number of nodes in the graph.")

    random_state = np.random.RandomState(seed) if seed is not None else np.random

    # Helper to compute minimum s-t cut for a given pair of nodes
    def compute_min_cut(source, target):
        cut_value, partition = nx.minimum_cut(graph, source, target, flow_func=nx.algorithms.flow.edmonds_karp)
        return cut_value, partition

    remaining_nodes = set(graph.nodes)
    partitions = []
    total_cut_value = 0

    while len(partitions) < k - 1:
        if len(remaining_nodes) < 2:
            break

        # Select two distinct nodes randomly
        source, target = random_state.choice(list(remaining_nodes), size=2, replace=False)
        cut_value, (reachable, non_reachable) = compute_min_cut(source, target)

        # Ensure partitions are disjoint
        reachable = reachable & remaining_nodes  # Intersect with remaining nodes
        non_reachable = non_reachable & remaining_nodes

        # Remove the smaller partition and keep the rest
        if len(reachable) <= len(non_reachable):
            partitions.append(reachable)
            remaining_nodes -= reachable
        else:
            partitions.append(non_reachable)
            remaining_nodes -= non_reachable

        total_cut_value += cut_value

    # Add the final partition
    partitions.append(remaining_nodes)

    return total_cut_value, partitions


def balanced_k_means(data, k, max_iter=100):
    """
    Perform balanced K-means clustering.
    
    Args:
        data (numpy.ndarray): The input data matrix (n_samples, n_features).
        k (int): The number of clusters.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        tuple: (cluster_assignments, cluster_centers)
    """
    n_samples = data.shape[0]
    cluster_size = n_samples // k
    remainder = n_samples % k

    # Initialize cluster centers randomly
    rng = np.random.default_rng()
    centers = data[rng.choice(n_samples, size=k, replace=False)]
    cluster_assignments = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        # Assign points to the nearest center
        new_assignments, _ = pairwise_distances_argmin_min(data, centers)
        
        # Balance clusters
        clusters = defaultdict(list)
        for idx, cluster in enumerate(new_assignments):
            clusters[cluster].append(idx)
        
        balanced_clusters = {i: [] for i in range(k)}
        remaining_indices = list(range(n_samples))
        
        for cluster in range(k):
            size = cluster_size + (1 if cluster < remainder else 0)
            while len(balanced_clusters[cluster]) < size:
                if not remaining_indices:
                    break
                distances = np.linalg.norm(data[remaining_indices] - centers[cluster], axis=1)
                closest_idx = np.argmin(distances)
                point_idx = remaining_indices.pop(closest_idx)
                balanced_clusters[cluster].append(point_idx)
        
        # Recompute cluster centers
        for cluster in range(k):
            indices = balanced_clusters[cluster]
            if indices:
                centers[cluster] = data[indices].mean(axis=0)
        
        # Check for convergence
        new_assignments = np.zeros(n_samples, dtype=int)
        for cluster, indices in balanced_clusters.items():
            for idx in indices:
                new_assignments[idx] = cluster
        
        if np.all(cluster_assignments == new_assignments):
            break
        
        cluster_assignments = new_assignments

    return cluster_assignments, centers
