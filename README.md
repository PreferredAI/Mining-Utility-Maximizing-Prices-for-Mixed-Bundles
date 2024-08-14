# Integrated Pricing of Overlapping Bundles via Optimizing Consumer Decisions

This paper examines a data-intensive optimization problem with an economically informed objective: given a set of bundles of items, we seek to price those bundles to maximize revenue. Going beyond existing work restricted to non-overlapping bundles, we address the more general scenario of overlapping bundles. This necessitates addressing two main challenges: determining the price of each bundle and calculating the revenue generated by a bundle at that price. Observing the mutual dependency between the two challenges, we propose the \textsc{Integrated Pricing Method}, a scalable approach that incorporates dependency-aware pricing. For the consumer demand problem, we motivate a principled utility maximization objective and solve it in terms of a weighted set packing formulation. In terms of bundle pricing, we introduce novel concepts such as competitive independence and dominance-based pruning. The effectiveness and efficiency of our approach are validated through experiments using real-world ratings-based datasets.

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/PreferredAI/integratedbundlepricing.git
```
2. Navigate to the project directory:
```bash
    cd integratedbundlepricing
```

3. Install the dependencies (requires Python 3.11.9)
```bash
    pip install -r requirements.txt
```

4. Navigate to the experiments directory:
```bash
    cd experiments
```

## Usage
To obtain the results in Table II, run
```bash
python experiment_baseline_comparison.py
```
To obtain the results in Figure 5, run 
```bash
python experiment_consumer_demand_problem.py
```
To obtain the results in Figure 6, run 
```bash
python experiment_component_bmkc_v_random.py
```
To obtain the results in Figure 7, run 
```bash
python experiment_component_bmkc_heuristic_scalability.py
```
To obtain the results in Figure 8, run 
```bash
python experiment_component_comp_ind_v_random.py
```
To obtain the results in Figure 9, run 
```bash
python experiment_component_lattice_v_graph.py
```
To obtain the results for pruning, run 
```bash
python experiment_component_pruning.py
```
To obtain the results in Figure 10, run 
```bash
python experiment_scalability_users.py 
```
and 
```bash
python experiment_scalability_bundles.py
```
