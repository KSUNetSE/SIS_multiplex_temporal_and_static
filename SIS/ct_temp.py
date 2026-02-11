# === Continuous-time (Gillespie) SIS on Static + Activity-Driven Layers ===
# - Static layer: BA-1000 from "BA-1000.txt" (adjlist)
# - Dynamic layer: Poisson activations (rate alpha) -> m instantaneous contacts
# - Infection on static S-I edges: rate beta (hazard per edge)
# - Recovery: rate mu_rate per infected node
# - Infection on dynamic contacts created at activation time: prob p_dyn = 1 - exp(-beta_dyn * delta_contact)

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

# ----------------------
# I/O + Network
# ----------------------
adjlist = 'ER-1000'
static_graph = nx.read_adjlist(f"{adjlist}.txt", nodetype=int)
N = static_graph.number_of_nodes()
static_edges_set = set(static_graph.edges())

# Undirected unique edge list (for fast S-I counting)
static_edges = list(static_graph.edges())
static_edges = [(u, v) if u < v else (v, u) for (u, v) in static_edges]
static_edges = list(set(static_edges))

# Candidates for dynamic (temporal) contacts: exclude static neighbors and self
candidate_neighbors = {}
for i in range(N):
    neighbors_i = set(static_graph.neighbors(i))
    candidates = [j for j in range(N) if j != i and j not in neighbors_i]
    candidate_neighbors[i] = candidates

# ----------------------
# Global params
# ----------------------
# Time + replication
T_horizon = 200.0            # CT simulated time (matches your discrete "timesteps")
obs_dt = 1.0                 # record every 1 unit -> series length int(T_horizon/obs_dt)+1
num_runs = 100
transient_cutoff = 100       # index in the recorded series (not CT time); must be <= T_horizon

# Epidemic
initial_infected_ratio = 0.01
mu_rate = 1.0                # recovery rate  (mean infectious time = 1)
alpha = 0.1                  # activation rate per node (Poisson)
m_list   = np.arange(6, 8, 1, dtype=int)      # m = 0..12
tau_list = np.round(np.arange(0.09, 0.1, 0.005), 3)  # 0.125..0.150 step 0.005

# Infection mapping: beta = tau * mu_rate
# You can also set beta_dyn != beta to weight dynamic contacts differently
beta_dyn_scale = 1.0         # 1.0 => same strength as static
delta_contact = 1.0          # effective duration for an instantaneous dynamic contact

# Output
csv_file = "sis_summary_results.csv"

# Threshold aides (static-only reference)
A = nx.to_numpy_array(static_graph, dtype=float)
eigvals = np.linalg.eigvals(A)
lambda_max_A = float(np.max(eigvals).real)
tau_c_static = 1.0 / lambda_max_A  # classical static SIS threshold (for m=0)

# ----------------------
# Helpers
# ----------------------
def _count_SI_static(states_dict):
    """Return (num_SI, SI_idx_list) for S-I edges on the static layer."""
    SI_idx = []
    for idx, (u, v) in enumerate(static_edges):
        su, sv = states_dict[u], states_dict[v]
        if (su == 'I' and sv == 'S') or (su == 'S' and sv == 'I'):
            SI_idx.append(idx)
    return len(SI_idx), SI_idx

def _choose_random_SI_edge(SI_idx):
    """Uniformly pick one S-I static edge index, return its (u, v)."""
    idx = random.choice(SI_idx)
    return static_edges[idx]

def run_sis_ct_once(m, tau, seed=None):
    """
    Continuous-time SIS with:
      - static infections at rate beta per S-I static edge,
      - recoveries at rate mu_rate per I node,
      - activations at rate alpha per node; an activation creates m instantaneous contacts.
    Returns dict with regular-sampled fractions {'S': list, 'I': list} at times 0,1,...,T_horizon.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    beta = float(tau * mu_rate)
    beta_dyn = float(beta_dyn_scale * beta)
    p_dyn = 1.0 - np.exp(-beta_dyn * delta_contact)

    # Init states
    states = {i: ('I' if random.random() < initial_infected_ratio else 'S') for i in range(N)}
    infected = {i for i, s in states.items() if s == 'I'}

    # Observation arrays (fractions)
    steps = int(T_horizon / obs_dt) + 1
    S_series = []
    I_series = []
    next_obs_t = 0.0

    def record_until(t_now):
        nonlocal next_obs_t
        while next_obs_t <= t_now and next_obs_t <= T_horizon:
            S_series.append((N - len(infected)) / N)
            I_series.append(len(infected) / N)
            next_obs_t += obs_dt

    t = 0.0
    record_until(0.0)

    # Gillespie loop
    while t < T_horizon:
        I = len(infected)
        if I == 0:
            record_until(T_horizon)
            break

        # rates
        R_rec = mu_rate * I
        num_SI, SI_idx = _count_SI_static(states)
        R_stat = beta * num_SI
        R_act = alpha * N
        R_tot = R_rec + R_stat + R_act

        if R_tot <= 0.0:
            record_until(T_horizon)
            break

        # waiting time
        u = random.random()
        dt = -np.log(u) / R_tot
        t += dt
        record_until(t)

        # choose event
        r = random.random() * R_tot
        if r < R_rec:
            # recovery: pick random infected
            node = random.choice(tuple(infected))
            states[node] = 'S'
            infected.remove(node)

        elif r < R_rec + R_stat and num_SI > 0:
            # static infection: pick random S-I edge; infect the S end
            u, v = _choose_random_SI_edge(SI_idx)
            if states[u] == 'I' and states[v] == 'S':
                states[v] = 'I'
                infected.add(v)
            elif states[v] == 'I' and states[u] == 'S':
                states[u] = 'I'
                infected.add(u)
            # if both already I or S due to earlier micro-step, nothing happens (rare)

        else:
            # activation: pick node uniformly (since all alpha equal)
            i = random.randrange(N)
            k = min(int(m), len(candidate_neighbors[i]))
            if k > 0:
                partners = random.sample(candidate_neighbors[i], k)
                # instantaneous contacts at time t
                for j in partners:
                    # if exactly one infected among (i, j), allow transmission with p_dyn
                    si_target = None
                    if states[i] == 'I' and states[j] == 'S':
                        si_target = j
                    elif states[i] == 'S' and states[j] == 'I':
                        si_target = i
                    if si_target is not None and random.random() < p_dyn:
                        states[si_target] = 'I'
                        infected.add(si_target)

    # pad (in case loop ended before final observation)
    if len(I_series) < steps:
        record_until(T_horizon)

    # Trim in case of rounding
    S_series = S_series[:steps]
    I_series = I_series[:steps]
    return {'S': S_series, 'I': I_series}

# ----------------------
# Main sweeps (no multiprocessing)
# ----------------------
if __name__ == "__main__":
    # Prepare/clear CSV
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=["Network", "m", "tau", "beta", "mu", "SteadyState"]).to_csv(csv_file, index=False)

    for m in m_list:
        for tau in tau_list:
            beta_here = float(tau * mu_rate)
            print(f"Running {num_runs} CT SIS runs for m={m}, tau={tau:.3f} (beta={beta_here:.3f}, mu={mu_rate})")

            # Run sequentially
            results = [run_sis_ct_once(m=m, tau=tau) for _ in range(num_runs)]

            # Stack results (already fractions!)
            all_S = np.array([r['S'] for r in results])
            all_I = np.array([r['I'] for r in results])

            # Averages + std
            mean_S = np.mean(all_S, axis=0)
            mean_I = np.mean(all_I, axis=0)
            std_S  = np.std(all_S, axis=0)
            std_I  = np.std(all_I, axis=0)

            # Simple steady-state proxy (same logic you had)
            tol = 0.05
            if transient_cutoff < len(mean_I):
                start_val = mean_I[transient_cutoff]
                end_val   = mean_I[-1]
            else:
                start_val = mean_I[-1]
                end_val   = mean_I[-1]

            if abs(start_val - end_val) < tol:
                sim_steady_state = 0.5 * (start_val + end_val)
            else:
                sim_steady_state = 0.0

            # First index within tolerance (optional)
            steady_indices = np.where(np.abs(mean_I - sim_steady_state) < tol)[0]
            steady_start = int(steady_indices[0]) if len(steady_indices) else transient_cutoff

            # --- Plot (save per (m, tau) to avoid overwrite)
            plot_steps = int(T_horizon)  # 200
            tgrid = np.arange(plot_steps)

            plt.figure(figsize=(10, 6))
            summary_text = (
                fr"$\tau$ = {tau:.3f}, $\beta$ = {beta_here:.3f}, $\mu$ = {mu_rate}, m = {m}" "\n"
                fr"SteadyState: {sim_steady_state:.4f}, steady_index: {steady_start}"
            )
            plt.gcf().text(
                0.02, 0.75, summary_text, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )
            plt.plot(tgrid, mean_S[:plot_steps], label='Susceptible', linewidth=2)
            plt.fill_between(tgrid, (mean_S - std_S)[:plot_steps], (mean_S + std_S)[:plot_steps], alpha=0.3)
            plt.plot(tgrid, mean_I[:plot_steps], label='Infected', linewidth=2)
            plt.fill_between(tgrid, (mean_I - std_I)[:plot_steps], (mean_I + std_I)[:plot_steps], alpha=0.3)
            plt.axhline(y=sim_steady_state, linestyle=':', linewidth=2, label=f'Simulated Steady State ({sim_steady_state:.2f})')
            plt.title(f"CT SIS (m={m}, tau={tau:.3f})")
            plt.xlabel("Time (CT units)")
            plt.ylabel("Fraction of Population")
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = f"sis_average_curve_m={m}_tau={tau:.3f}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved plot: {fname}")

            # --- Append CSV
            row = pd.DataFrame([{
                "Network": adjlist,
                "m": int(m),
                "tau": float(tau),
                "beta": beta_here,
                "mu": mu_rate,
                "SteadyState": round(float(sim_steady_state), 6),
            }])
            row.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"Appended summary to {csv_file}")
            print("======================")
