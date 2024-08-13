
import numpy as np
import itertools
import numpy as np

from functions.consumer_decision import *   



def conditional_price(whole_items, wtp, bundle, fixed_set, fixed_prices, theta, constant, pruning):
    # Find wtp_bundle
    wtp_bundle = np.sum(wtp[:, [whole_items.index(item) for item in bundle]], axis=1)  # Select relevant columns and sum
    max_surplus = np.sum(wtp_bundle)

    # Find the bundle unique prices
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))

    # Generate list of unique prices with indices from range(0, len(unique_prices), 5)
    conditional_prices = [unique_prices[i] for i in range(0, len(unique_prices), 5)]
    conditional_prices.append(1e6)
    

    temp_fixed_set_conditional = fixed_set.copy()
    temp_fixed_set_conditional.add(bundle)
    temp_fixed_prices_conditional = fixed_prices.copy()

    max_so_far = 0 # the total revenue
    conditional_pricing = None
    conditional_revenue_dict = None # the revenue breakdown
    for i in conditional_prices:
        temp_fixed_prices_conditional[bundle] = i
        cond_revenue, revenue_dict = calculate_revenue(whole_items, wtp, temp_fixed_set_conditional, temp_fixed_prices_conditional)
        if cond_revenue >= max_so_far:
            max_so_far = cond_revenue
            conditional_pricing = i
            conditional_revenue_dict = revenue_dict
    
    prune = None
    # Pruning enabled 
    if pruning:
        # dominance check
        if conditional_revenue_dict[bundle] < theta * max_surplus: # if the bundle gets less than theta max surplus
            fixed_set_subsets = set([subset for subset in temp_fixed_set_conditional if subset.issubset(bundle)])
            fixed_set_subsets_prices = {bundle: price for bundle, price in temp_fixed_prices_conditional.items() if bundle in fixed_set_subsets}

            pruning_max_so_far = 0 # the total revenue
            pruning_conditional_price = None
            pruning_conditional_revenue_dict = None # the revenue breakdown
            for i in conditional_prices:
                fixed_set_subsets_prices[bundle] = i
                cond_revenue, revenue_dict = calculate_revenue(whole_items, wtp, fixed_set_subsets, fixed_set_subsets_prices)
                if cond_revenue >= pruning_max_so_far:
                    pruning_max_so_far = cond_revenue
                    pruning_conditional_price = i
                    pruning_conditional_revenue_dict = revenue_dict

            prune = pruning_conditional_revenue_dict[bundle] < theta * max_surplus


    return conditional_pricing, max_so_far, prune


# vertex is the one we're caculating the CI_score for
def competitive_price(whole_items, wtp, vertex, wtp_vertex, vertex_rmp, vertex_std_dev, bundle, fixed_set, fixed_prices, constant):
    # Find wtp_bundle
    wtp_bundle = np.sum(wtp[:, [whole_items.index(item) for item in bundle]], axis=1)  # Select relevant columns and sum

    # Find the bundle unique prices
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))

    # Generate list of unique prices with indices from range(0, len(unique_prices), 5)
    bundle_unique_prices = [unique_prices[i] for i in range(0, len(unique_prices), 5)]
    bundle_unique_prices.append(1e6)
    
    # Find the vertex unique prices
    sorted_vertex_wtp = sorted(wtp_vertex)
    v_unique_prices = sorted(set(sorted_vertex_wtp))

    # Generate list of unique prices with indices from range(0, len(unique_prices), 5)
    vertex_unique_prices = [v_unique_prices[i] for i in range(0, len(v_unique_prices), 5)]
    vertex_unique_prices.append(1e6)
    
    price_pairs = list(itertools.product(bundle_unique_prices, vertex_unique_prices))

    temp_fixed_set_competitive = fixed_set.copy()
    temp_fixed_set_competitive.add(bundle)
    temp_fixed_set_competitive.add(vertex)

    temp_fixed_prices_competitive = fixed_prices.copy()

    max_so_far = 0 # the total revenue
    competitive_price = None
    competitive_revenue_dict = None # the revenue breakdown

    for pair in price_pairs:
        bundle_price, vertex_price = pair

        temp_fixed_prices_competitive[vertex] = vertex_price
        temp_fixed_prices_competitive[bundle] = bundle_price
        
        comp_revenue, revenue_dict = calculate_revenue(whole_items, wtp, temp_fixed_set_competitive, temp_fixed_prices_competitive) # OPTIMISE TO remove the recalculation of the WTP

        if comp_revenue >= max_so_far:
            max_so_far = comp_revenue
            competitive_price = pair
            competitive_revenue_dict = revenue_dict

    bundle_price, vertex_price = competitive_price

    return bundle_price, vertex_price, max_so_far, competitive_revenue_dict


def price_refinement(whole_items, wtp, bundle, bundle_price, fixed_set, fixed_prices, constant):
    
    # Find wtp_bundle
    wtp_bundle = np.sum(wtp[:, [whole_items.index(item) for item in bundle]], axis=1)  # Select relevant columns and sum

    # Find the bundle unique prices
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))

    # Generate list of unique prices with indices from range(0, len(unique_prices), 5)
    conditional_prices = [unique_prices[i] for i in range(0, len(unique_prices), 5)]
    conditional_prices.append(1e6)


    temp_fixed_set_conditional = fixed_set.copy()
    temp_fixed_prices_conditional = fixed_prices.copy()

    max_so_far = 0 # the total revenue
    conditional_pricing = None
    conditional_revenue_dict = None # the revenue breakdown
    for i in conditional_prices:
        temp_fixed_prices_conditional[bundle] = i
        cond_revenue, revenue_dict = calculate_revenue(whole_items, wtp, temp_fixed_set_conditional, temp_fixed_prices_conditional)
        if cond_revenue >= max_so_far:
            max_so_far = cond_revenue
            conditional_pricing = i
            conditional_revenue_dict = revenue_dict
    
    return conditional_pricing, conditional_revenue_dict
    
