from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import networkx as nx
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from collections import defaultdict

def spectral_clustering_min_k_cut(graph, k):
    """
    Perform spectral clustering to solve the balanced minimum k-cut problem.
    
    Parameters:
    graph (nx.Graph): An undirected weighted graph.
    k (int): The number of clusters.
    
    Returns:
    list: A list of lists, where each sublist contains the nodes in one of the k clusters.
    """
    # Step 1: Compute the graph Laplacian
    laplacian = nx.laplacian_matrix(graph).astype(float)
    
    # Step 2: Compute the first k eigenvectors of the Laplacian
    _, eigvecs = eigsh(laplacian, k=k, which='SM')
    
    # Step 3: Use k-means to cluster the nodes based on the eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(eigvecs)
    
    # Step 4: Form the partitions
    partitions = [[] for _ in range(k)]
    for node, cluster_id in zip(graph.nodes(), clusters):
        partitions[cluster_id].append(node)
    
    return partitions


def balanced_spectral_clustering_min_k_cut(graph, k):
    """
    Perform balanced spectral clustering to solve the balanced minimum k-cut problem.

    Parameters:
    graph (nx.Graph): An undirected weighted graph.
    k (int): The number of clusters.

    Returns:
    list: A list of lists, where each sublist contains the nodes in one of the k clusters.
    """
    # Step 1: Map graph nodes to integers
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    # Step 2: Compute the graph Laplacian
    laplacian = nx.laplacian_matrix(graph).astype(float)

    # Step 3: Compute the first k eigenvectors of the Laplacian
    _, eigvecs = eigsh(laplacian, k=k, which='SM')

    # Step 4: Use k-means to cluster the nodes based on the eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=0)
    initial_clusters = kmeans.fit_predict(eigvecs)

    # Step 5: Balance the clusters
    # Calculate the desired size for each cluster
    num_nodes = len(graph.nodes())
    cluster_size = num_nodes // k
    remainder = num_nodes % k

    clusters = defaultdict(list)
    for node, cluster_id in zip(graph.nodes(), initial_clusters):
        clusters[cluster_id].append(node)

    balanced_clusters = {i: [] for i in range(k)}
    remaining_indices = list(range(num_nodes))

    for cluster in range(k):
        size = cluster_size + (1 if cluster < remainder else 0)
        while len(balanced_clusters[cluster]) < size:
            if not remaining_indices:
                break
            distances = [
                (idx, np.linalg.norm(eigvecs[idx] - kmeans.cluster_centers_[cluster]))
                for idx in remaining_indices
            ]
            distances.sort(key=lambda x: x[1])
            for idx, _ in distances:
                if len(balanced_clusters[cluster]) < size:
                    balanced_clusters[cluster].append(index_to_node[idx])
                    remaining_indices.remove(idx)
                else:
                    break

    # Convert to final partitions
    partitions = [balanced_clusters[i] for i in range(k)]

    return partitions


def calculate_cut_value(G, partition):
    cut_value = 0
    for i, cluster1 in enumerate(partition):
        for cluster2 in partition[i+1:]:
            for node1 in cluster1:
                for node2 in cluster2:
                    if G.has_edge(node1, node2):
                        cut_value += G[node1][node2]['weight']
    return cut_value