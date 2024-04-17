import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sc

supply = 1 # total nutrient supply
delta = supply 
e = 1 # total enzyme budget

steps = 5 # paper uses 1200

d = 1 # diffusion coefficient
l = 20 # length of the system
s = np.array([.25, .45, .3]) # nutrient supply
p = len(s) # number of nutrients
np.random.seed(28)


m = 10 # number of species
def randomStrats(number):
    # ? Make this explicitly pulling from simplex if doesn't work
    return np.random.dirichlet(np.ones(p), number)
alpha = randomStrats(m)


# TODO: purpose, included in paper?
# alpha[2, 1] += .1
# alpha[2, 0] -= .1
# alpha[5] = np.array([.6, .2, .2])
# alpha = np.insert(alpha, 7, np.array([.7, .1, .2]), axis=0)
# m = 11

# Nutrient Diffusion functions

strat = alpha[:, 0] # maps strategies to continuum of nutrient 1
sScaled = (e/np.sum(s))*s[0] # Normalized nutrient supply


def getPositions(pop0):
    m0 = len(pop0)
    extendedPop = np.insert(pop0, 0, 0)
    return np.array([[np.sum(extendedPop[:i-1]), np.sum(extendedPop[:i])] for i in range(2, m0+2)])

def getConcentration(n, nutrient): 
    m0 = len(n)
    aVars = [sp.symbols(f'a{k}') for k in range(m0)]
    bVars = [sp.symbols(f'b{k}') for k in range(m0)]
    def c(a, b, alpha, theta):
        return a*np.exp(np.sqrt(alpha/d)*theta) + b*np.exp(-np.sqrt(alpha/d)*theta) + s[nutrient]/alpha
    def dc(a, b, alpha, theta):
        return np.sqrt(alpha/d)*(a*np.exp(np.sqrt(alpha/d)*theta) - b*np.exp(-np.sqrt(alpha/d)*theta))

    # Create a system of eqs for c(\[Theta]) in each region
    # Enforces that c(\[Theta]) is continuous at the boundaries
    # These are half of the requirements on A and B
    cSys = [c(aVars[i], bVars[i], alpha[i, nutrient], n[i]) - c(aVars[i+1], bVars[i+1], alpha[i+1, nutrient], 0) for i in range(m0-1)]
    cSysWrap = c(aVars[m0-1], bVars[m0-1], alpha[m0-1, nutrient], n[m0-1]) - c(aVars[0], bVars[0], alpha[0, nutrient], 0)
    cSystem = cSys + [cSysWrap]

    # do the same thing for the derivatives of c(\[Theta])
    # These too must be continuous at the boundaries
    # These are the other half of the requirements on A and B
    dcSys = [dc(aVars[i], bVars[i], alpha[i, nutrient], n[i]) - dc(aVars[i+1], bVars[i+1], alpha[i+1, nutrient], 0) for i in range(m0-1)]
    dcSysWrap = dc(aVars[m0-1], bVars[m0-1], alpha[m0-1, nutrient], n[m0-1]) - dc(aVars[0], bVars[0], alpha[0, nutrient], 0)
    dcSystem = dcSys + [dcSysWrap]

    totalcSys = cSystem + dcSystem
    print(totalcSys)
    exit()
    # vars = Table[{aVars[[i]], bVars[[i]]}, {i, m0}] // Flatten;
    vars = [item for i in zip(aVars, bVars) for item in i]
    # TODO: make numpy to be faster
    # m, z = sp.linear_eq_to_matrix(totalcSys, vars)
    solution = sp.solve(totalcSys, vars)
    # print(solution)
    # exit()
    return np.array([solution[v] for v in vars])


def fullConcentration(n, nutrient):
    coeff = getConcentration(n, nutrient)
    positions = getPositions(n)
    a = coeff[::2]
    b = coeff[1::2]
    def c(a, b, alpha, theta):
        return a*np.exp(np.sqrt(alpha/d)*theta) + b*np.exp(-np.sqrt(alpha/d)*theta) + s[nutrient]/alpha
    pieces = [c(a[j], b[j], alpha[j, nutrient], sp.symbols("theta") - positions[j, 0]) for j in range(m)]
    conditions = [positions[i, 0] < sp.symbols("theta") <= positions[i, 1] for i in range(m)]
    return np.piecewise(sp.symbols("theta"), conditions, pieces)


# Population Dynamics Functions: 

def getGrowth(pop, species, c):
    myN = pop[species]
    myA = c[species*2] # 0/1 indexing? Should this be species*2? Similarly for the next: + 1?
    myB = c[species*2 + 1]
    growth = np.array([alpha[species, i]*((s[i]*myN/alpha[species, i]) + np.sqrt(d/alpha[species, i])*(myA[i]*(np.exp(myN*np.sqrt(alpha[species, i]/d)) - 1) - myB[i]*(np.exp(-myN*np.sqrt(alpha[species, i]/d)) - 1))) for i in range(p)]).sum()
    return growth


def ndot(n, t=0):
    n0 = n
    print(n0)
    cCoeff = np.array([getConcentration(n0, v) for v in range(p)]).T
    print(cCoeff.shape)
    exit()
    growthTable = np.array([getGrowth(n0, i, cCoeff) for i in range(len(n0))])
    dn = (growthTable - delta*n0).astype(np.float64)
    print(f"Working: {t}")
    return dn

# Initial Conditions and Integration

nInit = l*np.ones(m)/m # initial population


# May need to modify signature of ndot?
sol = sc.integrate.odeint(ndot, nInit, np.linspace(0, steps, steps+1))

exit()

# np.savetxt("sol2.txt", sol)

# sol = np.loadtxt("sol2.txt") 

# Plotting

# plt.figure()
# plt.plot(sol/l)
# plt.yscale('log')
# plt.ylim(10**-3, 1)
# plt.xlim(0, steps)
# plt.xticks([0, 300, 600, 900, 1200], [0, 300, 600, 900, 1200])
# plt.yticks([.001, .01, .1, 1], [.001, .01, .1, 1])
# plt.show()


sampleTimes = np.arange(0, 76, 1)
numSamples = len(sampleTimes)

popSamples = np.array([sol[t] for t in sampleTimes])

positionSamples = np.array([getPositions(pop) for pop in popSamples])

# TODO: check this function
boundarySamples = np.array([[positionSamples[i, j, 1] for j in range(m)] for i in range(numSamples)])

pairListTimeOrder = np.array([np.transpose([boundarySamples[i], np.ones(m)*sampleTimes[i]]) for i in range(numSamples)])

pairListSpeciesOrder = np.transpose(pairListTimeOrder, (1, 0, 2))


flippedPairListSpeciesOrder = np.array([[pairListSpeciesOrder[i, j][::-1] for j in range(numSamples)] for i in range(m)])
print(flippedPairListSpeciesOrder.shape)


# fillingChart = {i: [[i-1], f'C{i}'] for i in range(2, m)}
# fillingChart[1] = [[0], 'C1']

plt.figure()
plt.fill_between(np.arange(numSamples), flippedPairListSpeciesOrder[0, :, -1], color=f'C0')
for spec in range(1, m):
    plt.fill_between(np.arange(numSamples), flippedPairListSpeciesOrder[spec-1, :, -1], flippedPairListSpeciesOrder[spec, :, -1], color=f'C{spec}')
plt.show()

