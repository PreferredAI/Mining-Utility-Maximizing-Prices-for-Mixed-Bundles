from functions.pricing_functions import *
# BASELINES

def no_competition_pricing(whole_items, bundle, wtp):
    # assume that the item sets are equal
    wtp_bundle = [0 for k in range(len(wtp))]
    for i in bundle:
        item_index = whole_items.index(i)
        for k in range(len(wtp)):
            wtp_bundle[k] += wtp[k][item_index]

    max_so_far = 0
    max_price = 0
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))
    for i in range(0, len(unique_prices), 5):  # Iterate through every 5 unique prices
        revenue = unique_prices[i] * (len(wtp_bundle) - sorted_wtp.index(unique_prices[i]))
        if revenue >= max_so_far:
            max_so_far = revenue
            max_price = unique_prices[i]
    return max_price, max_so_far

def rmp_baseline(whole_items, wtp, bundles):
    bundle_prices = {bundle: 1e6 for bundle in bundles}
    for bundle in bundles:
        bundle_price, _ = no_competition_pricing(whole_items, bundle, wtp)
        bundle_prices[bundle] = bundle_price
    
    total_revenue, revenue_dict = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    return total_revenue


def rmp_baseline_return_price(whole_items, wtp, bundles):
    bundle_prices = {bundle: 1e6 for bundle in bundles}
    for bundle in bundles:
        bundle_price, _ = no_competition_pricing(whole_items, bundle, wtp)
        bundle_prices[bundle] = bundle_price
    
    total_revenue, revenue_dict = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    return total_revenue, bundle_prices

