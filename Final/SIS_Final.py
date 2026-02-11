import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

# === Parameters ===
adjlist = 'graphs/WS-1000.adjlist'
static_graph = nx.read_adjlist(f"{adjlist}", nodetype=int)

# adjlist = '../BA-1000'
# static_graph = nx.read_adjlist(f"{adjlist}.txt", nodetype=int)

N = static_graph.number_of_nodes()
static_edges_set = set(static_graph.edges())

# m_values = [0, 10]
activity = 0.1
activity_rates = {node: activity for node in static_graph.nodes}

timesteps = 200
num_runs = 100

# beta = 0.19
mu = 1

m_list   = np.arange(0, 16, 1)                     # m sweep
tau_list = np.round(np.arange(0.16, 0.32, 0.005), 3)  # tau sweep

transient_cutoff = 100
initial_infected_ratio = 0.1

csv_file = "WS_HM.csv"


# A = nx.to_numpy_array(static_graph, dtype=int)
# eigenvalues, _ = np.linalg.eig(A)
# lambda_max_A = np.max(eigenvalues)
# tau_c = 1 / lambda_max_A
# tau = beta/mu
# m_c = ((1 / tau) - (lambda_max_A)) / (2*activity)

candidate_neighbors = {}
for i in range(N):
    candidates = [j for j in range(N) if j != i and (i, j) not in static_edges_set and (j, i) not in static_edges_set]
    candidate_neighbors[i] = candidates

# === Run Simulation Function ===
def run_sis_once(args):
    # m = args[0]
    m, beta = args
    states = {i: 'I' if random.random() < initial_infected_ratio else 'S' for i in range(N)}
    counts = {'S': [], 'I': []}

    for _ in range(timesteps):
        G_temporal = nx.Graph()
        G_temporal.add_nodes_from(range(N))
        for i in range(N):
            if random.random() < activity_rates[i]:
                targets = random.sample(candidate_neighbors[i], min(m, len(candidate_neighbors[i])))
                G_temporal.add_edges_from((i, j) for j in targets)

        infected_set = {n for n, s in states.items() if s == 'I'}
        new_states = states.copy()

        for node in range(N):
            if states[node] == 'I':
                if random.random() < mu:
                    new_states[node] = 'S'
            elif states[node] == 'S':
                infected_neighbors = 0

                for neighbor in static_graph.neighbors(node):
                    if neighbor in infected_set:
                        infected_neighbors += 1

                for neighbor in G_temporal.neighbors(node):
                    if neighbor in infected_set:
                        infected_neighbors += 1

                if infected_neighbors > 0:
                    p_infection = 1 - (1 - beta) ** infected_neighbors
                    if random.random() < p_infection:
                        new_states[node] = 'I'

        states = new_states
        counts['S'].append(sum(1 for s in states.values() if s == 'S'))
        counts['I'].append(sum(1 for s in states.values() if s == 'I'))

    return counts

if __name__ == "__main__":
    # for m in range (m_values[0], m_values[1] + 1):
    for m in m_list:
        for tau in tau_list:
            beta = tau * mu
            # === Run Multiple Simulations ===
            print(f"Running {num_runs} SIS simulations for m = {m} and tau = {beta}...")

            # with ProcessPoolExecutor() as executor:
            #     results = list(executor.map(run_sis_once, [(m,)] * num_runs))

            args_iter = [(m, beta) for _ in range(num_runs)]
            with ProcessPoolExecutor() as ex:
                results = list(ex.map(run_sis_once, args_iter))

            all_S = [r['S'] for r in results]
            all_I = [r['I'] for r in results]


            # === Convert to numpy arrays ===
            all_S = np.array(all_S) / N
            all_I = np.array(all_I) / N

            # === Compute Averages and StdDev ===
            mean_S = np.mean(all_S, axis=0)
            mean_I = np.mean(all_I, axis=0)
            std_S = np.std(all_S, axis=0)
            std_I = np.std(all_I, axis=0)

            # === Calculate Simulated Steady State ===
            # sim_steady_state = np.mean(mean_I[transient_cutoff:])
            tolerance = 0.05
            start_val = mean_I[transient_cutoff]
            end_val = mean_I[-1]
            if abs(start_val - end_val) < tolerance:
                sim_steady_state = (start_val + end_val) / 2
            else:
                sim_steady_state = 0.0
            
            tolerance = 0.05  # tuned based on noise
            steady_indices = np.where(np.abs(mean_I - sim_steady_state) < tolerance)[0]
            if len(steady_indices) > 0:
                steady_start = steady_indices[0]
            else:
                steady_start = transient_cutoff


            summary_data = {
                'Network': adjlist,
                'm': [m],
                'beta': [beta],
                'mu': [mu],
                'SteadyState': [f"{sim_steady_state:.4f}"],
            }
            df = pd.DataFrame(summary_data)

            if os.path.exists(csv_file):
                existing_df = pd.read_csv(csv_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df

            combined_df.to_csv(csv_file, index=False)
            print(f"Saved summary to {csv_file}")
            print("======================")
