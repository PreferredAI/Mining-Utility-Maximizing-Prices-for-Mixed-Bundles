import numpy as np
import math
import random
from time import process_time
import numpy as np
from functions.consumer_decision import *
from functions.pricing_functions import *
from functions.lattice import *
from functions.algorithm import *
from functions.split_heuristic import *
from functions.baseline import *
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import random
from functions.cluster import *
from functions.min_k_cut import *


def overall_heuristic_split(whole_items, complete, wtp, clusters, refinement, prune, bmkc, comp_ind, lattice, constant = 0.15, random_CI = False, theta = 0.1, partition_method = 'random', alpha = 1, beta = 1):
    
    # split the complete into many bundles, and run the pricing algo
    bundles = list(complete)

    start = process_time()
    if bmkc:
        # Lattice
        graph = build_lattice(bundles)
        avg_size = math.ceil(len(bundles) / clusters) 

        partitions = []
        # Split, heuristic
        while len(graph.nodes) > 0:
            partition = partition_lattice_heuristic(graph, avg_size)

            partitions.append(partition)


    elif partition_method == 'random': 
        
        # Random split implmenetation
        complete_list = list(complete)
        # Shuffle the order of bundles randomly
        random.shuffle(complete_list)

        # Split the shuffled bundles into clusters
        cluster_size = math.ceil(len(complete_list) / clusters) 
        partitions = [set(complete_list[i:i+cluster_size]) for i in range(0, len(complete_list), cluster_size)]


    elif partition_method == 'kmeans':
         # Generate the binary matrix
        matrix = generate_bundle_matrix(bundles, whole_items)
        cluster_assignments, cluster_centers = k_means_clustering(matrix, clusters)

        # Group bundles by their cluster assignments
        clusterss = {i: [] for i in range(clusters)}
        for idx, cluster in enumerate(cluster_assignments):
            clusterss[cluster].append(bundles[idx])

        # Convert clusters dictionary to a list of clusters
        clusters_list = [clusterss[i] for i in range(clusters)]

        partitions = clusters_list

    elif partition_method == 'balanced_kmeans':
         # Generate the binary matrix
        matrix = generate_bundle_matrix(bundles, whole_items)
        cluster_assignments, cluster_centers = balanced_k_means(matrix, clusters)

        # Group bundles by their cluster assignments
        clusterss = {i: [] for i in range(clusters)}
        for idx, cluster in enumerate(cluster_assignments):
            clusterss[cluster].append(bundles[idx])

        # Convert clusters dictionary to a list of clusters
        clusters_list = [clusterss[i] for i in range(clusters)]

        partitions = clusters_list


    elif partition_method == 'minkcut':
        graph = build_lattice(bundles)

        cut_value, partitions = min_k_cut(graph, clusters, seed=42)
        
    elif partition_method == 'spectral':
        
        # Simple Graph
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    simple_graph.add_edge(bundle1, bundle2, weight = len(bundle1.intersection(bundle2)))

        spectral_partitions = spectral_clustering_min_k_cut(simple_graph, clusters)

        partitions = spectral_partitions
    
    elif partition_method == 'balanced_spectral':
        
        # Simple Graph
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    simple_graph.add_edge(bundle1, bundle2, weight = len(bundle1.intersection(bundle2)))

        spectral_partitions = balanced_spectral_clustering_min_k_cut(simple_graph, clusters)

        partitions = spectral_partitions
    end = process_time()

    # print([len(i) for i in partitions])
    # return 0, 0 

    complete_prices = {}

    # PRICE
    for partition in partitions:
        cluster_price = initialize_algorithm(whole_items, wtp, partition, pruning=prune, comp_ind=comp_ind, lattice=lattice, constant=constant, random_CI=random_CI, theta = theta, alpha=alpha, beta=beta)
        complete_prices.update(cluster_price)

    revenue, revenue_breakdown = calculate_revenue(whole_items, wtp, complete, complete_prices)


    # REFINEMENT
    if refinement:
        sorted_bundles = sorted(revenue_breakdown.keys(), key=lambda bundle: revenue_breakdown[bundle])
        for bundle in sorted_bundles:

            refined_pricing, refined_revenue_dict = price_refinement(whole_items, wtp, bundle, complete_prices[bundle], complete.copy(), complete_prices.copy(), constant)
            complete_prices[bundle] = refined_pricing

        refined_revenue, refined_revenue_breakdown = calculate_revenue(whole_items, wtp, complete, complete_prices)

        return refined_revenue, revenue
    else:
        return 0, revenue



def overall_heuristic_split_irefine(whole_items, complete, wtp, clusters, refinement, prune, bmkc, comp_ind, lattice, constant = 0.15, random_CI = False, theta =0.1, partition_method = 'random', alpha = 1, beta = 1):
    
    const = 1.0003
    # split the complete into many bundles, and run the pricing algo
    bundles = list(complete)

    start = process_time()
    if bmkc:
        # Lattice
        graph = build_lattice(bundles)
        avg_size = math.ceil(len(bundles) / clusters) 

        partitions = []
        # Split, heuristic
        while len(graph.nodes) > 0:
            partition = partition_lattice_heuristic(graph, avg_size)

            partitions.append(partition)

    elif partition_method == 'random': 
        
        # Random split implmenetation
        complete_list = list(complete)
        # Shuffle the order of bundles randomly
        random.shuffle(complete_list)

        # Split the shuffled bundles into clusters
        cluster_size = math.ceil(len(complete_list) / clusters) 
        partitions = [set(complete_list[i:i+cluster_size]) for i in range(0, len(complete_list), cluster_size)]


    elif partition_method == 'kmeans':
         # Generate the binary matrix
        matrix = generate_bundle_matrix(bundles, whole_items)
        cluster_assignments, cluster_centers = k_means_clustering(matrix, clusters)

        # Group bundles by their cluster assignments
        clusterss = {i: [] for i in range(clusters)}
        for idx, cluster in enumerate(cluster_assignments):
            clusterss[cluster].append(bundles[idx])

        # Convert clusters dictionary to a list of clusters
        clusters_list = [clusterss[i] for i in range(clusters)]

        partitions = clusters_list

    elif partition_method == 'minkcut':
        graph = build_lattice(bundles)

        cut_value, partitions = min_k_cut(graph, clusters, seed=42)
        
    elif partition_method == 'spectral':
        
        # Simple Graph
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    simple_graph.add_edge(bundle1, bundle2, weight = len(bundle1.intersection(bundle2)))

        spectral_partitions = spectral_clustering_min_k_cut(simple_graph, clusters)

        partitions = spectral_partitions
        
    elif partition_method == 'balanced_spectral':
        
        # Simple Graph
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    simple_graph.add_edge(bundle1, bundle2, weight = len(bundle1.intersection(bundle2)))

        spectral_partitions = balanced_spectral_clustering_min_k_cut(simple_graph, clusters)

        partitions = spectral_partitions

    elif partition_method == 'balanced_kmeans':
         # Generate the binary matrix
        matrix = generate_bundle_matrix(bundles, whole_items)
        cluster_assignments, cluster_centers = balanced_k_means(matrix, clusters)

        # Group bundles by their cluster assignments
        clusterss = {i: [] for i in range(clusters)}
        for idx, cluster in enumerate(cluster_assignments):
            clusterss[cluster].append(bundles[idx])

        # Convert clusters dictionary to a list of clusters
        clusters_list = [clusterss[i] for i in range(clusters)]

        partitions = clusters_list
        
    end = process_time()

    total_coverage = np.sum(wtp)

    complete_prices = {}

    # PRICE
    for partition in partitions:
        cluster_price = initialize_algorithm(whole_items, wtp, partition, pruning=prune, comp_ind=comp_ind, lattice=lattice, constant=constant, random_CI=random_CI, theta = theta, alpha=alpha, beta=beta)
        complete_prices.update(cluster_price)

    revenue, revenue_breakdown = calculate_revenue(whole_items, wtp, complete, complete_prices)
    old_complete_prices = complete_prices.copy()
    old_revenue = revenue

    # REFINEMENT
    r = 0
    if refinement:
        sorted_bundles = sorted(revenue_breakdown.keys(), key=lambda bundle: revenue_breakdown[bundle])
        for bundle in sorted_bundles:

            refined_pricing, refined_revenue_dict = price_refinement(whole_items, wtp, bundle, complete_prices[bundle], complete.copy(), complete_prices.copy(), constant)
            complete_prices[bundle] = refined_pricing

        refined_revenue, refined_revenue_breakdown = calculate_revenue(whole_items, wtp, complete, complete_prices)


        nocomp_bundle_prices = {bundle: 1e6 for bundle in complete}
        for bundle in complete:
            bundle_price, _ = no_competition_pricing(whole_items, bundle, wtp)
            nocomp_bundle_prices[bundle] = bundle_price
    
        ncp_revenue, revenue_dict = calculate_revenue(whole_items, wtp, complete, nocomp_bundle_prices)

        if ncp_revenue > refined_revenue:
            complete_prices = nocomp_bundle_prices.copy()
            refined_revenue = ncp_revenue
            refined_revenue_breakdown = revenue_dict.copy()



    if refinement:
        while True:
            r += 1
                
            sorted_bundles = sorted(refined_revenue_breakdown.keys(), key=lambda bundle: refined_revenue_breakdown[bundle])
            for bundle in sorted_bundles:

                refined_pricing, refined_revenue_dict = price_refinement(whole_items, wtp, bundle, complete_prices[bundle], complete.copy(), complete_prices.copy(), constant)
                complete_prices[bundle] = refined_pricing

            refined_revenue, refined_revenue_breakdown = calculate_revenue(whole_items, wtp, complete, complete_prices)

            if refined_revenue < old_revenue * const:
                break
            old_revenue = refined_revenue

        return refined_revenue, revenue
    else:
        return 0, revenue

