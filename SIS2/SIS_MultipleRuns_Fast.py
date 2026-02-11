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
adjlist = 'BA-1000'
static_graph = nx.read_adjlist(f"../{adjlist}.txt", nodetype=int)
N = static_graph.number_of_nodes()
static_edges_set = set(static_graph.edges())

# m_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# m_values = [0, 20]
m = 10
activity = 0.2
activity_values = [20, 20]


timesteps = 500
num_runs = 500

beta = 0.09
mu = 0.5

# beta = 0.11
# mu = 1

transient_cutoff = 100
initial_infected_ratio = 0.15

csv_file = "sis_summary_results.csv"

# def calculate_steady_state_probabilities(A, tau, max_iter=100000, tol=1e-6):
#     N = len(A)
#     y = np.ones(N) * 100

#     for _ in range(max_iter):
#         denom = 1 + y
#         y_new = tau * (A @ (y / denom))

#         if np.allclose(y, y_new, rtol=tol, atol=tol):
#             break
#         y = y_new

#     p = y / (1 + y)
#     return p

# adj_matrix = nx.to_numpy_array(static_graph)
# tau = beta / mu
# p = calculate_steady_state_probabilities(adj_matrix, tau)
# theo_steady_state = np.mean(p)


candidate_neighbors = {}
for i in range(N):
    candidates = [j for j in range(N) if j != i and (i, j) not in static_edges_set and (j, i) not in static_edges_set]
    candidate_neighbors[i] = candidates


# === Run Simulation Function ===
def run_sis_once(args):
    m = args[0]
    activity = args[1]
    activity_rates = {node: activity for node in static_graph.nodes}
    # activity = args[0]
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
    for activity in range (activity_values[0], activity_values[1] + 1):
        activity = activity * 0.005
        # === Run Multiple Simulations ===
        # print(f"Running {num_runs} SIS simulations for m = {m}...")
        print(f"Running {num_runs} SIS simulations for a = {activity}...")
    
        

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_sis_once, [(m, activity)] * num_runs))

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
        tolerance = 0.01
        start_val = mean_I[transient_cutoff]
        end_val = mean_I[-1]
        if abs(start_val - end_val) < tolerance:
            sim_steady_state = (start_val + end_val) / 2
        else:
            sim_steady_state = 0.0

        
        tolerance = 0.01  # tuned based on noise
        steady_indices = np.where(np.abs(mean_I - sim_steady_state) < tolerance)[0]
        if len(steady_indices) > 0:
            steady_start = steady_indices[0]
        else:
            steady_start = transient_cutoff

        # === Plot ===
        plt.figure(figsize=(10, 6))
        plot_steps = 200
        t = np.arange(plot_steps)

        summary_text = (
            f"$\\beta$ = {beta}, $\\mu$ = {mu}, m = {m}\n"
            # f"theo_SteadyState: {theo_steady_state:.2f}\n"
            f"activity: {activity:.3f}\n"
            f"SteadyState: {sim_steady_state:.4f}\n"
            f"steady_index: {steady_start}\n"
        )

        # plt.gcf().text(0.02, 0.75, summary_text, fontsize=10, verticalalignment='top',
        #                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        plt.rcParams['axes.titlesize'] = 20       # Title
        plt.rcParams['axes.labelsize'] = 18       # X/Y labels
        plt.rcParams['xtick.labelsize'] = 16      # Tick labels
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16      # Legend


        plt.plot(t, mean_S[:plot_steps], label='Susceptible', color='skyblue', linewidth=2)
        plt.fill_between(t, mean_S[:plot_steps] - std_S[:plot_steps], mean_S[:plot_steps] + std_S[:plot_steps], color='skyblue', alpha=0.3)

        plt.plot(t, mean_I[:plot_steps], label='Infected', color='red', linewidth=2)
        plt.fill_between(t, mean_I[:plot_steps] - std_I[:plot_steps], mean_I[:plot_steps] + std_I[:plot_steps], color='red', alpha=0.3)

        # plt.axhline(y=theo_steady_state, color='green', linestyle='--', linewidth=2, label=f'Theoretical Steady State ({theo_steady_state:.2f})')
        plt.axhline(y=sim_steady_state, color='purple', linestyle=':', linewidth=2, label=f'Steady State Prevalence')

        # plt.title(f"Average SIS (activity = {activity:.3f})")
        plt.xlabel("Time Step")
        plt.ylabel("Fraction of Population")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"sis_average_curve_activity={activity:.3f}.png", dpi=300)
        plt.close()

        print(f"Saved average SIS curve plot as sis_average_curve_activity={activity:.3f}.png")

        summary_data = {
            'Network': adjlist,
            'activity': activity,
            'm': [m],
            'beta': [beta],
            'mu': [mu],
            # 'theo_SteadyState': [theo_steady_state],
            'SteadyState': [f"{sim_steady_state:.4f}"],
            'steady_index:': [steady_start],
        }
        df = pd.DataFrame(summary_data)

        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df

        # combined_df.to_csv(csv_file, index=False)
        # print(f"Saved summary to {csv_file}")
        print("======================")
