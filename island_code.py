import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy import interpolate

import fractal

def plot_fig1(state, N, m):
    normState = state / N
    mean_x = np.mean(normState)

    bins = 100

    xes = np.linspace(0, 1, 1000)
    beta_ys = stats.beta.pdf(xes, 2 * N * m * mean_x, 2 * N * m * (1 - mean_x))

    plt.hist(normState, bins=bins)
    plt.plot(xes, beta_ys * len(normState) / bins, color='red')
    plt.show()
    exit()

def wf(s, m, N, D):
    states = []
    state = np.array([N/2] * D)

    tfix = 0
    while np.sum(state) != 0 and np.sum(state) != N * D:
        states.append(state)

        x = state / N
        mean_x = np.mean(x)
        print(mean_x)
        
        if mean_x > 0.611 or mean_x < 0.389:
            plot_fig1(state, N, m)
        
        s_freq = s * (x - 1/2)

        p = (1 - m) * x + m * mean_x
        print(max(p))
        p_vec = (1 + s_freq) * p / (1 + s_freq * p)

        # print(p_vec)
        for i in range(D):
            new = np.random.binomial(N, p_vec[i])
            state[i] = new

        tfix += 1
    
    states.append(state)

    return np.sum(state), tfix, states

def wf_custom_m(s, ms, N, D):
    assert(len(ms) == D)
    state = np.array([N/2] * D)

    tfix = 0
    while np.sum(state) != 0 and np.sum(state) != N * D:
        x = state / N
        mean_x = np.mean(x)
        
        # if mean_x > 0.611 or mean_x < 0.389:
        #     plot_fig1(state, N, m)
        
        s_freq = s * (x - 1/2)

        p = np.clip((1 - ms) * x + sum(ms * x) / D, 0, 1)
        p_vec = (1 + s_freq) * p / (1 + s_freq * p)

        # print(p_vec)
        for i in range(D):
            new = np.random.binomial(N, p_vec[i])
            state[i] = new

        tfix += 1
    
    return np.sum(state), tfix


def network_diffusion_steady(G, pop_state, alpha, S, D):
    L = nx.laplacian_matrix(G).toarray()

    alpha_state = np.zeros(len(pop_state))
    for idx in range(0, len(alpha)):
        alpha_state += (pop_state == idx) * alpha[idx]

    a = np.add(D * L, np.diag(alpha_state))
    b = np.array([S] * len(pop_state))

    C = np.linalg.solve(a, b)

    return C, alpha_state


def network_fit(G, pop_state, alpha, S, D):
    f = np.zeros(len(pop_state))
    for idx, alpha_sel in enumerate(alpha):
        C, alpha_state = network_diffusion_steady(G, pop_state, alpha_sel, S[idx], D[idx])
        f = np.add(f, np.multiply(C, alpha_state))

    return f

if __name__ == "__main__":

    # Parameters
    N = 100 # Number of islands
    D = 2000 # Number of demes
    m = 0.01 # Migration rate
    s = 0.001 # Selection coefficient
    trials = 1 # Number of trials

    # m_fract_its = 8
    # m_fract_dim = 0.5

    # inter_xs, ys = fractal.makeFractal(m_fract_its, m_fract_dim, 0.025)
    # f = interpolate.interp1d(inter_xs, ys)

    # xs = np.linspace(0, 1, D)
    # ms = np.array([f(x) for x in xs])
    # print(ms)

    # for _ in range(trials):
    #     final, t = wf_custom_m(s, ms, N, D)
    #     print(final)
    
    for _ in range(trials):
        final, t, states = wf(s, m, N, D)
        print(final)

    # alpha = [0.1, 0.2, 0.3]
    # S = [1, 1, 1]
    # D = [1, 1, 1]

    # # Generate population
    # pop = np.zeros((N, D))
    # for i in range(N):
    #     pop[i, :] = np.random.randint(0, 3, D)

    # # Generate network
    # G = nx.erdos_renyi_graph(N, 0.1)

    # # Run simulation
    # tfix = 0
    # for i in range(1000):
    #     pop = np.random.permutation(pop)
    #     for j in range(N):
    #         pop[j, :], tfix_temp = wf(s, m, N, D)
    #         tfix += tfix_temp

    # print(tfix)
    # print(network_fit(G, pop, alpha, S, D))