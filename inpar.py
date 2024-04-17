import numpy as np
import networkx as nx
import random
from itertools import count, filterfalse
from matplotlib import pyplot as plt
from multiprocessing import Pool
import sys

# G: networkx weighted graph
# s: selection coefficient
# mu: mutation rate
# max_gen: maximum number of generations
# rich: number of unqiue species

def G_mut(G, s, mu, max_gen):
    A = nx.to_numpy_array(G)
    degs = np.sum(A, 0)
    D_inv = np.array(degs, dtype=float) ** -1
    AD_inv = np.multiply(A, D_inv).T

    state = {}
    N = len(G)

    for idx in range(0, N):
        state[idx] = 0
    init_n = int(N / 2)
    for idx in random.sample(range(0, N), init_n):
        state[idx] = 1

    rich = []

    tfix = 0
    while tfix < max_gen:
        x = np.zeros(N)
        for i in np.unique(list(state.values())):
            tmp_states = np.array(list(state.values())) == i
            mut_x = AD_inv @ tmp_states
            x += np.multiply(mut_x, tmp_states)

        weights = 0.5 + s * (x - 0.5)
        node_sel = random.choices(list(state.keys()), weights=weights)

        nn = list(G.neighbors(node_sel[0]))
        weights = AD_inv[[node_sel[0]], nn]
        nn_sel = random.choices(nn, weights=weights)
        state[nn_sel[0]] = state[node_sel[0]]

        if random.random() < mu:
            new = next(filterfalse(set(list(state.values())).__contains__, count(1)))
            node_sel = random.randint(0, N - 1)
            state[node_sel] = new

        rich.append(len(np.unique(list(state.values()))))

        tfix += 1

    return rich

def isl(G_list, delta):

    G = nx.Graph()
    for sub_G in G_list:
        G = nx.disjoint_union(G, sub_G.copy())

    e_list = list(G.edges())
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1

    N = len(G)
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in e_list and (j, i) not in e_list:
                G.add_edge(i, j)
                G[i][j]['weight'] = delta

    return G

def run_G_mut(d):
    print(d)
    r2 = G_mut(isl([nx.cycle_graph(50), nx.cycle_graph(50)], d), -1, 0.01, 10000)
    return np.mean(r2[100:])


if __name__ == "__main__":
    print(run_G_mut(sys.argv[1]))

# G = isl([nx.cycle_graph(50), nx.cycle_graph(50)], )