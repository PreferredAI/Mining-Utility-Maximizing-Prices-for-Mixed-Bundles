from functions.lattice import *



# this gets one partition, then redraws the lattice accordingly. 
def partition_lattice_heuristic(graph, avg_size):
    partition = set()

    if len(graph.nodes) <= avg_size:
        # Add all the graph nodes into partition
        partition.update(graph.nodes)
            
        # Remove all nodes from the graph
        graph.remove_nodes_from(list(graph.nodes))

        return partition


    while len(graph.nodes) > 0 and len(partition) < avg_size:
        # Randomly choose the largest bundle as the anchor bundle
        sorted_nodes = sorted(graph.nodes, key=len, reverse=True)
        for anchor_bundle in sorted_nodes:
            if anchor_bundle not in partition:
                # Add anchor_bundle to the partition
                partition.add(anchor_bundle)
                break


        # Add subsets of the anchor bundle until partition size reaches avg_size
        subsets = [node for node in graph.successors(anchor_bundle) if node not in partition] 
        while len(partition) < avg_size and subsets:

            subset = max(subsets, key=len)
            subsets.remove(subset)
            partition.add(subset)


            supersets = [node for node in graph.predecessors(subset) if node not in partition]

            # Add supersets of the subset until partition size reaches avg_size
            while len(partition) < avg_size and supersets:

                superset = max(supersets, key=len)
                supersets.remove(superset)
                partition.add(superset)

    # Sort partition with the largest bundles first
    sorted_partition = sorted(partition, key=len, reverse=True)
    for bundle in sorted_partition:
        redraw_lattice(graph, bundle)
    

    return partition