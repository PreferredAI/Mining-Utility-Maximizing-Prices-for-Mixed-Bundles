from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import networkx as nx


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



def calculate_cut_value(G, partition):
    cut_value = 0
    for i, cluster1 in enumerate(partition):
        for cluster2 in partition[i+1:]:
            for node1 in cluster1:
                for node2 in cluster2:
                    if G.has_edge(node1, node2):
                        cut_value += G[node1][node2]['weight']
    return cut_value