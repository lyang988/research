import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

gen = np.random.default_rng(5182003)

def set_seed(seed):
    global gen
    gen = np.random.default_rng(seed)
    
def split(ys : list[float], displacementFactor : float) -> list[float]:
    newYs = []

    for i in range(len(ys) - 1):
        midpoint = (ys[i] + ys[i + 1]) / 2.0
        displacement = gen.random() * displacementFactor
        newYs.append(ys[i])
        newYs.append(midpoint + displacement)
    
    newYs.append(ys[-1])

    return newYs

def randomSampler(xs, ys):
    cumulative = np.cumsum(ys)
    f = interpolate.interp1d(cumulative/cumulative.max(), xs)
    def sampler():
        return f(gen.random()) 
    return sampler

def makeFractal(iterations, roughness, displacementFactor):
    ys = [0.0, 0.0]

    for i in range(iterations):
        ys = split(ys, displacementFactor)
        displacementFactor *= roughness
    xs = np.linspace(0, 1, len(ys))

    return xs, ys


if __name__ == '__main__':
    iterations = 10
    roughness = .5 # Changes number of bumps 
    displacementFactor = 5.0 # Changes max y
    
    roughnesses = [ 0, .2, .4, .6, .8]

    fig, axes = plt.subplots(5, 1)
    for i, r in enumerate(roughnesses):
        xs, ys = makeFractal(iterations, r, displacementFactor)
        axes[i].plot(xs, ys)

    plt.show()
