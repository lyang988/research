import numpy as np
import networkx as nx
import random
from itertools import count, filterfalse
from matplotlib import pyplot as plt
import fractal
from scipy import interpolate
import itertools

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
        if tfix % 10000 == 0:
            print(tfix)

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
    isl_G = isl([nx.cycle_graph(50), nx.cycle_graph(50)], d)
    r2 = G_mut(isl_G, -1, 0.01, 100 * 10000)

    # its = list(nx.get_edge_attributes(isl_G,'weight').items())
    # its.sort(key=lambda x: x[1])
    # edges, weights = zip(*its)
    # print(weights)
    # nx.draw(isl_G, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues)
    # plt.show()
    return np.mean(r2[100 * 100:])


def interpolate_xs_ys(xs, ys, new_xs):
    f = interpolate.interp1d(xs, ys)
    return f(new_xs)


def generate_graph_1(xs, ys, thresh=2/3, delta=0.005):
    G = nx.Graph()
    for i in range(len(xs)):
        G.add_node(i)
        if i > 0:
            # Adds all 'adjacent' edges
            G.add_edge(i, i - 1)
            G[i][i - 1]['weight'] = 1
    
    y_thresh = max(ys) * thresh
    at_least_thresh = [n for n in range(len(xs)) if ys[n] >= y_thresh]
    for (n, m) in itertools.combinations(at_least_thresh, 2):
        G.add_edge(n, m)
        G[n][m]['weight'] = 1

    nx.draw(G, pos={n: (xs[n], real_frac_ys[n]) for n in G.nodes()})
    plt.show()

    for ne in nx.non_edges(G):
        G.add_edge(ne[0], ne[1])
        G[ne[0]][ne[1]]['weight'] = delta

    return G


def generate_graph_2(xs, ys, delta=0.005):
    # Line of sight

    G = nx.Graph()
    for i in range(len(xs)):
        G.add_node(i)
    
    for n in range(len(xs)):
        max_tan = -np.inf
        for m in range(n + 1, len(xs)):
            tan = (ys[m] - ys[n]) / (xs[m] - xs[n])
            if tan >= max_tan:
                G.add_edge(n, m)
                G[n][m]['weight'] = 1
                max_tan = tan

    nx.draw(G, pos={n: (xs[n], real_frac_ys[n]) for n in G.nodes()})
    plt.show()

    for ne in nx.non_edges(G):
        G.add_edge(ne[0], ne[1])
        G[ne[0]][ne[1]]['weight'] = delta

    return G


if __name__ == "__main__":
    xs = np.linspace(0, 1, 100)
    delta = 0.005
    threshold_fraction = 2/3

    fractal.set_seed(5182003)
    roughnesses = [0, .2, .4, .6, .8]
    fractals = []
    for r in roughnesses:
        if r == 0:
            frac_xs, frac_ys = xs, np.zeros(len(xs))
        else:
            frac_xs, frac_ys = fractal.makeFractal(10, r, 1)
        real_frac_ys = interpolate_xs_ys(frac_xs, frac_ys, xs)
        fractals.append(generate_graph_1(xs, real_frac_ys, threshold_fraction, delta))
        fractals.append(generate_graph_2(xs, real_frac_ys, delta))

    # 
    
    # reses = []
    # for G in fractals:
    #     r2 = G_mut(G, -1, 0.01, 100 * 10000)
    #     reses.append(np.mean(r2[100 * 100:]))
    
    # print(roughnesses, reses)
    # plt.plot(roughnesses, reses)
    # plt.xlabel("Roughness")
    # plt.ylabel("Richness")
    # plt.show()

# G = isl([nx.cycle_graph(50), nx.cycle_graph(50)], )