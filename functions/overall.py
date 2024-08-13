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

def overall_heuristic_split(whole_items, complete, wtp, clusters, refinement, prune, bmkc, comp_ind, lattice, constant = 0.15, random_CI = False, theta = 0.0001):
    
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

    else: 
        # Random split implmenetation
        complete_list = list(complete)
        # Shuffle the order of bundles randomly
        random.shuffle(complete_list)

        # Split the shuffled bundles into clusters
        cluster_size = math.ceil(len(complete_list) / clusters) 
        partitions = [set(complete_list[i:i+cluster_size]) for i in range(0, len(complete_list), cluster_size)]
    end = process_time()


    complete_prices = {}

    # PRICE
    for partition in partitions:
        cluster_price = initialize_algorithm(whole_items, wtp, partition, pruning=prune, comp_ind=comp_ind, lattice=lattice, constant=constant, random_CI=random_CI, theta = theta)
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



def overall_heuristic_split_irefine(whole_items, complete, wtp, clusters, refinement, prune, bmkc, comp_ind, lattice, constant = 0.15, random_CI = False):
    
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

    else: 
        # Random split implmenetation
        complete_list = list(complete)
        # Shuffle the order of bundles randomly
        random.shuffle(complete_list)

        # Split the shuffled bundles into clusters
        cluster_size = math.ceil(len(complete_list) / clusters) 
        partitions = [set(complete_list[i:i+cluster_size]) for i in range(0, len(complete_list), cluster_size)]
    end = process_time()

    total_coverage = np.sum(wtp)

    complete_prices = {}

    # PRICE
    for partition in partitions:
        cluster_price = initialize_algorithm(whole_items, wtp, partition, pruning=prune, comp_ind=comp_ind, lattice=lattice, constant=constant, random_CI=random_CI)
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

