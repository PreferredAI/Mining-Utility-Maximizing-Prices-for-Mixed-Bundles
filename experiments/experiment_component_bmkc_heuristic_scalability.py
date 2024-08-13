# test heuristic scalability compared to baseline methods for min k cut problem; 

import numpy as np
import argparse
import math
import random
import warnings
import itertools
warnings.filterwarnings("error")

import sys
sys.path.append("..")
from functions.pricing_functions import *
from functions.algorithm import *
from functions.alternative_consumer_decision import *
from functions.baseline import *
from functions.consumer_decision import *
from functions.overall import *
from functions.min_k_cut import *
import networkx as nx



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default =30)
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-s", "--size", type=int, help="Size", default = 5)
parser.add_argument("-p", "--proportion", type=float, help="Proportion", default = 0.5)
parser.add_argument("-cls", "--clustersize", type=int, help="Cluster Size", default = 5)
parser.add_argument("-m", "--maxscale", type=int, help="Max Scale", default = 10)
args = parser.parse_args()


seed = args.seed
random.seed(seed)
repetitions = args.repetitions
size = args.size
proportion = args.proportion
cluster_size = args.clustersize
max_scale = args.maxscale


data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
whole_items = [i for i in data_PID]
np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")

# Random 100 consumers 
random_indices = np.random.choice(np_wtp.shape[0], size=100, replace=False)
np_wtp = np_wtp[random_indices]

bundle_sizes = []


# Define the format string
format_string = "{:<25}{:<25}{:<30}{:<30}{:<30}"


print("Scalability - Heuristic Compared to Spectral Clustering")
# Print the headers
print(format_string.format("Number of Bundles (B)", 
                           "Heuristic Time", 
                           "Spectral Clustering Time", 
                           "Heuristic Cut Value", 
                           "Spectral Clustering Cut Value"))

# Initialize the results dictionary
comparison_results = { 'overall_time': []}

scale_start_time = process_time()
for scale in range(max_scale):
    scale_end_time = process_time()

    overall_time_list = []
    heuristic_cut = []
    lib_time_list = []
    spectral_cut = []

    # Initialize the results dictionary
    for _ in range(repetitions):
        target_items = random.sample(whole_items, size + scale)
        indices = [whole_items.index(i) for i in target_items]

        n1 = np.arange(len(np_wtp))
        temp_wtp = np.round(np_wtp[n1[:,None],indices], 2)
        temp_whole_items = target_items.copy()

        total_coverage = np.sum(temp_wtp)


        # Generate powerset
        powerset = set()
        for r in range(2, len(target_items) + 1):  # Start from 2 to exclude single-item sets
            for combination in itertools.combinations(target_items, r):
                powerset.add(frozenset(tuple(combination)))

        # Sample proportion p
        num_to_sample = int(len(powerset) * proportion)
        sampled_set = set(random.sample(list(powerset), num_to_sample))  # Convert set to list

        # Add single sets
        for item in target_items:
            sampled_set.add(frozenset({item}))

        bundles = sampled_set
        
        clusters = int(len(bundles) / cluster_size)

        start = process_time()
        bundle_list = list(bundles)
        graph = build_lattice(bundle_list)
        graph_copy = graph.copy()
        avg_size = math.ceil(len(bundles) / clusters) 
        
        partitions = []
        # Split, heuristic
        while len(graph.nodes) > 0:
            partition = partition_lattice_heuristic(graph, avg_size)

            partitions.append(partition)
        
        end = process_time()
        
        overall_time_list.append(end - start)



        start = process_time()

        # Simple Graph
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(bundles)

        # Add edges between bundles if they have at least one item in common
        for bundle1 in bundles:
            for bundle2 in bundles:
                if bundle1 != bundle2 and bundle1.intersection(bundle2):
                    simple_graph.add_edge(bundle1, bundle2, weight = len(bundle1.intersection(bundle2)))
        spectral_partitions = spectral_clustering_min_k_cut(simple_graph, clusters)
        end = process_time()
        lib_time_list.append(end - start)

        heuristic_cut_value = calculate_cut_value(simple_graph, partitions)
        heuristic_cut.append(heuristic_cut_value)

        spectral_cut_value = calculate_cut_value(simple_graph, spectral_partitions)
        spectral_cut.append(spectral_cut_value)

        
    bundle_sizes.append(len(bundles))
    # Print the values aligned with headers
    print(format_string.format(len(bundles),
                            f"{np.mean(overall_time_list):.4e}",
                            f"{np.mean(lib_time_list):.4e}",
                            f"{np.mean(heuristic_cut):.4e}",
                            f"{np.mean(spectral_cut):.4e}"))

