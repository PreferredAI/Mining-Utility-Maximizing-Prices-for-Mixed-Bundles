
import numpy as np
import random
from time import process_time
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np
from functions.consumer_decision import *
from functions.pricing_functions import *
from functions.lattice import *

def initialize_algorithm(whole_items, wtp, bundles, pruning, constant, comp_ind, lattice, theta = 0.0001, random_CI = False):


    # CONSTANTS
    theta = theta
    CONSTANT = constant
    alpha = 1  
    beta = 1   
    
    candidate_set = set(bundles)

    # Initialize variables
    fixed_set = set()
    fixed_prices = {bundle:1e6 for bundle in bundles}
    
    start = process_time()
    # Initialize the graph
    if lattice:
        graph = build_lattice(candidate_set)
    else:
        # Simple Graph
        graph = nx.Graph()
        graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    graph.add_edge(bundle1, bundle2)
    end = process_time()


    conditional_price_time = 0
    competitive_price_time = 0
    redraw_lattice_time = 0

    pricing_seq = []


    total_start = process_time()
    # Main loop while 𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒 is not empty
    while candidate_set:
        overall_conditional_prices = {bundle: 1e6 for bundle in candidate_set}

        # Loop over each vertex in J
        for vertex in list(graph.nodes()):
            # add into overall_cond_price
            conditional_start = process_time()
            
            conditional_pricing, revenue, prune = conditional_price(whole_items, wtp, vertex, fixed_set, fixed_prices, theta, CONSTANT, pruning)
            overall_conditional_prices[vertex] = conditional_pricing

            conditional_end = process_time()
            conditional_price_time += conditional_end - conditional_start


            if prune:
                redraw_start = process_time()
                
                if lattice:
                    redraw_lattice(graph, vertex)
                else:
                    graph.remove_node(vertex)
                
                redraw_end = process_time()
                redraw_lattice_time += redraw_end - redraw_start

                candidate_set.remove(vertex)

                # Break if candidate_set is empty
                if not candidate_set:
                    break

        # Break if candidate_set is empty (Break out of while loop)
        if not candidate_set:
            break

        # Initialize competitive_independence_score dictionary
        competitive_independence_score = {bundle: float('inf') for bundle in candidate_set}

        for vertex in list(graph.nodes()):
            # Initialize an empty list of competitive prices divided by the corresponding conditional price
            competitive_prices_ratio = []
            
            # calculate the bundle wtp so we dont repeat calculations later
            wtp_vertex, vertex_rmp, vertex_std_dev = calculate_bundle_wtp(whole_items, wtp, vertex)
            
            # consider case with no neighbors, simply price using conditional price and remove from the candidate set
            if lattice:
                neighbors_list = list(set(graph.neighbors(vertex)) | set(graph.predecessors(vertex)))
            else:
                neighbors_list = list(set(graph.neighbors(vertex)))

            if len(neighbors_list) == 0:
                fixed_prices[vertex] = overall_conditional_prices[vertex]

                # Remove the corresponding vertex from the graph
                graph.remove_node(vertex)

                # Add the vertex to fixed_set
                fixed_set.add(vertex)

                # Remove the bundle from the candidate_set
                candidate_set.remove(vertex)
                continue

            # Iterate through the neighbors of the vertex
            for neighbor in neighbors_list:
                competitive_start = process_time()

                # Calculate competitive price for the neighbor; 
                _, vertex_competitive_price, competitive_revenue, competitive_revenue_dict = competitive_price(whole_items, wtp, vertex, wtp_vertex, vertex_rmp, vertex_std_dev, neighbor, fixed_set, fixed_prices, CONSTANT)
                vertex_competitive_price /= overall_conditional_prices[vertex]

                competitive_end = process_time()
                competitive_price_time += competitive_end - competitive_start

                competitive_prices_ratio.append(vertex_competitive_price)
            
            # Calculate mean and standard deviation of competitive_prices list
            mean_price = np.mean(competitive_prices_ratio)
            std_dev_price = np.std(competitive_prices_ratio) 
            
            # Calculate CI_score
            CI_score = alpha * (1 - mean_price)**2 + beta * std_dev_price**2
            
            # Update the dict with the new CI_score
            competitive_independence_score[vertex] = CI_score
        

        # Find the key with the highes value in competitive_independence_score
        if comp_ind:
            min_vertex = max(competitive_independence_score, key=competitive_independence_score.get)

        else:
            # Random
            if random_CI:
                min_vertex = random.choice(list(competitive_independence_score.keys()))
            else:
            # Lowest value CI_score
                min_vertex = min(competitive_independence_score, key=competitive_independence_score.get)

        pricing_seq.append(min_vertex)
        if min_vertex in graph:
            redraw_start = process_time()

            if lattice:
                redraw_lattice(graph, min_vertex)
            else:
                graph.remove_node(min_vertex)

            redraw_end = process_time()
            redraw_lattice_time += redraw_end - redraw_start

            # Add the vertex to fixed_set
            fixed_set.add(min_vertex)

            # Remove the bundle from the candidate_set
            candidate_set.remove(min_vertex)

        # Assign the price to fixed_prices
        fixed_prices[min_vertex] = overall_conditional_prices[min_vertex]

    total_end = process_time()

    return fixed_prices

