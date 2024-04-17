import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

gen = np.random.default_rng(9282003)
    
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


if __name__ == '__main__':
    iterations = 8
    roughness = .3 # Changes bumpiness
    displacementFactor = 1.0 # Changes max y
    ys = [0.0, 0.0]

    for i in range(iterations):
        ys = split(ys, displacementFactor)
        displacementFactor *= roughness
    xs = np.linspace(0, 1, len(ys))

    generator = randomSampler(xs, ys)
    samples = [generator() for _ in range(10000)]
    fig, axes = plt.subplots(2, 2)
    # fig, axes = plt.subplots(2, 2)
    axes[0,0].hist(samples, bins=100)
    axes[0,1].plot(xs, ys)

    iterations = 8
    roughness = .5 # Changes bumpiness
    displacementFactor = 1.0 # Changes max y
    ys = [0.0, 0.0]

    for i in range(iterations):
        ys = split(ys, displacementFactor)
        displacementFactor *= roughness
    xs = np.linspace(0, 1, len(ys))

    generator = randomSampler(xs, ys)
    samples = [generator() for _ in range(10000)]
    # fig, axes = plt.subplots(1, 2)
    # fig, axes = plt.subplots(2, 2)
    axes[1,0].hist(samples, bins=100)
    axes[1,1].plot(xs, ys)
    plt.show()
