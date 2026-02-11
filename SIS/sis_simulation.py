import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# === Parameters ===
beta = 0.002           # Infection rate
delta = 0.01          # Recovery rate
a = 0.1               # Activation probability
m = 5                 # Number of links each active node creates
T = 200               # Number of time steps
initial_infected_frac = 0.5  # Initial infection

# === Load static layer ===
static_graph = nx.read_adjlist("../WS-1000.txt", nodetype=int)
nodes = list(static_graph.nodes())
N = len(nodes)

# === State indices ===
# 0: S1 (susceptible + inactive)
# 1: S2 (susceptible + active)
# 2: I1 (infected + inactive)
# 3: I2 (infected + active)

# === Initialization ===
states = np.zeros((N, 4))  # One-hot vector per node

# Randomly infect a subset
infected_indices = np.random.choice(N, size=int(initial_infected_frac * N), replace=False)
for i in range(N):
    if i in infected_indices:
        states[i, 2] = 1  # Start in I1
    else:
        states[i, 0] = 1  # Start in S1

# === Tracking ===
prevalence = []

# === Simulation Loop ===
for t in range(T):
    next_states = np.zeros((N, 4))
    active_nodes = []
    temp_links = {i: [] for i in range(N)}

    # 1. Activation
    for i in range(N):
        if random.random() < a:
            active_nodes.append(i)

    # 2. Generate temporal links (undirected)
    for i in active_nodes:
        partners = random.sample([j for j in range(N) if j != i], m)
        for j in partners:
            temp_links[i].append(j)
            temp_links[j].append(i)

    # 3. Compute infection pressure
    I2 = states[:, 3]  # Infected + active
    lambda_1 = np.zeros(N)
    lambda_2 = np.zeros(N)

    for i in range(N):
        # Static layer
        for j in static_graph.neighbors(i):
            j = int(j)
            lambda_1[i] += beta * (states[j, 2] + states[j, 3])

        # Temporal layer
        for j in temp_links[i]:
            lambda_2[i] += beta * states[j, 3]

    # 4. Update state transitions
    for i in range(N):
        S1, S2, I1, I2 = states[i]

        # --- Update S1
        p_stay = S1 * (1 - a - lambda_1[i])
        p_from_S2 = S2 * (1 - lambda_2[i]) * (1 - a)
        p_from_I1 = I1 * delta
        next_states[i, 0] = p_stay + p_from_S2 + p_from_I1

        # --- Update S2
        p_from_S1 = a * S1 * (1 - lambda_1[i] - lambda_2[i])
        p_stay_S2 = S2 * (1 - lambda_1[i] - lambda_2[i]) * (1 - a)
        next_states[i, 1] = p_from_S1 + p_stay_S2

        # --- Update I1
        next_states[i, 2] = I1 * (1 - delta - a) + I2 * (1 - delta) * (1 - a)

        # --- Update I2
        next_states[i, 3] = a * I1 * (1 - delta) + I2 * (1 - delta) * (1 - a)

    states = next_states

    # 5. Track prevalence
    prevalence.append(np.sum(states[:, 2] + states[:, 3]) / N)

# === Plot ===
plt.plot(prevalence, label="Infected Fraction")
plt.xlabel("Time")
plt.ylabel("Prevalence")
plt.title("SIS on Static + Activity-driven Network")
plt.grid()
plt.legend()
plt.show()
