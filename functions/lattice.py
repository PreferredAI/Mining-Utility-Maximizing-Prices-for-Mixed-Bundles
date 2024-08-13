# Lattice functions
import networkx as nx
from collections import defaultdict


# Function to build a lattice structure from a given set of bundles; 
def build_lattice(bundles):
    
    # Initialize the graph
    graph = nx.DiGraph()
    
    # Group bundles by their sizes
    size_to_bundles = defaultdict(list)
    for bundle in bundles:
        size_to_bundles[len(bundle)].append(bundle)

    # Add bundles as nodes to the graph
    graph.add_nodes_from(bundles)
    
    # Find pairs of bundles based on size difference and add edges
    max_size = max(len(bundle) for bundle in bundles)
    for i in range(1, max_size + 1):
        for size_small in range(1, max_size + 1 - i):
            size_large = size_small + i
            bundles_small = size_to_bundles[size_small]
            bundles_large = size_to_bundles[size_large]
            
            for bundle_small in bundles_small:
                for bundle_large in bundles_large:
                    if bundle_small.issubset(bundle_large):
                        if i > 1:
                            if not nx.has_path(graph, bundle_large, bundle_small):
                                graph.add_edge(bundle_large, bundle_small)
                        else:
                            graph.add_edge(bundle_large, bundle_small)


    return graph

# Redraws lattice based on the removal of one node
def redraw_lattice(graph, node):

    supersets = set(graph.predecessors(node))
    subsets = set(graph.successors(node))

    # Remove
    graph.remove_node(node)
    # Redraw lattice
    # For each superset-subset pair, check if a path exists
    for superset in supersets:
        for subset in subsets:
            if not nx.has_path(graph, superset, subset):
                graph.add_edge(superset, subset)


