import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

supply = 1 # total nutrient supply
delta = supply 
e = 1 # total enzyme budget

steps = 1200 # paper uses 1200

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
    abVarMatrix = np.zeros((2 * m0, 2 * m0))
    finalMatrix = np.zeros((2 * m0, 1))
    bVarOffset = m0
    def cA(alpha, theta):
        # Coefficient of A in C at position specified by theta
        return np.exp(np.sqrt(alpha/d)*theta)
    def cB(alpha, theta):
        # Coefficient of B in C at position specified by theta
        return np.exp(-np.sqrt(alpha/d)*theta)
    def cC(alpha):
        # Constant term in C at position
        return s[nutrient]/alpha
    
    def dcA(alpha, theta):
        # Coefficient of A in derivative of C (wrt theta) at position specified by theta
        return np.sqrt(alpha/d)*np.exp(np.sqrt(alpha/d)*theta)
    def dcB(alpha, theta):
        # Coefficient of B in derivative of C (wrt theta) at position specified by theta
        return -np.sqrt(alpha/d)*np.exp(-np.sqrt(alpha/d)*theta)
    def dcC():
        # Constant term in derivative of C (wrt theta) at position
        return 0

    for i in range(m0):
        nextIndex = (i + 1) % m0

        # First half of the requirements on A and B: c(\[Theta]) is continuous at the boundaries
        abVarMatrix[i, i] = cA(alpha[i, nutrient], n[i])
        abVarMatrix[i, i + bVarOffset] = cB(alpha[i, nutrient], n[i])
        abVarMatrix[i, nextIndex] = -cA(alpha[nextIndex, nutrient], 0)
        abVarMatrix[i, nextIndex + bVarOffset] = -cB(alpha[nextIndex, nutrient], 0)
        finalMatrix[i] -= cC(alpha[i, nutrient])
        finalMatrix[i] += cC(alpha[nextIndex, nutrient]) # These are 'flipped' because we 'move' the terms to the other side of the equation

        # Second half of the requirements on A and B: the derivatives of c(\[Theta]) are continuous at the boundaries
        abVarMatrix[i + m0, i] = dcA(alpha[i, nutrient], n[i])
        abVarMatrix[i + m0, i + bVarOffset] = dcB(alpha[i, nutrient], n[i])
        abVarMatrix[i + m0, nextIndex] = -dcA(alpha[nextIndex, nutrient], 0)
        abVarMatrix[i + m0, nextIndex + bVarOffset] = -dcB(alpha[nextIndex, nutrient], 0)
        finalMatrix[i + m0] -= dcC()
        finalMatrix[i + m0] += dcC() # These are 'flipped' because we 'move' the terms to the other side of the equation

    # print(abVarMatrix)
    # print(finalMatrix)
    # exit()

    solution = np.linalg.solve(abVarMatrix, finalMatrix)
    # print(solution)
    # exit()
    return solution


def fullConcentration(n, nutrient):
    coeff = getConcentration(n, nutrient)
    positions = getPositions(n)
    m0 = len(n)
    a = coeff[:m0]
    b = coeff[m0:]
    def c(a, b, alpha, theta):
        return a*np.exp(np.sqrt(alpha/d)*theta) + b*np.exp(-np.sqrt(alpha/d)*theta) + s[nutrient]/alpha
    pieces = [c(a[j], b[j], alpha[j, nutrient], sp.symbols("theta") - positions[j, 0]) for j in range(m)]
    conditions = [positions[i, 0] < sp.symbols("theta") <= positions[i, 1] for i in range(m)]
    return np.piecewise(sp.symbols("theta"), conditions, pieces)


# Population Dynamics Functions: 

def getGrowth(m0, pop, species, c):
    myN = pop[species]
    myA = c[0, species] # 0/1 indexing? Should this be species*2? Similarly for the next: + 1?
    myB = c[0, m0 + species]
    growth = np.array([alpha[species, i]*((s[i]*myN/alpha[species, i]) + np.sqrt(d/alpha[species, i])*(myA[i]*(np.exp(myN*np.sqrt(alpha[species, i]/d)) - 1) - myB[i]*(np.exp(-myN*np.sqrt(alpha[species, i]/d)) - 1))) for i in range(p)]).sum()
    return growth


def ndot(n, t=0):
    n0 = n
    # print(n0)
    m0 = len(n0)
    print(n)
    cCoeff = np.array([getConcentration(n0, v) for v in range(p)]).T
    # print(cCoeff.shape)
    # exit()
    growthTable = np.array([getGrowth(m0, n0, i, cCoeff) for i in range(m0)])
    dn = (growthTable - delta*n0).astype(np.float64)
    print(f"Working: {t}")
    return dn

# Initial Conditions and Integration

nInit = l*np.ones(m)/m # initial population


# May need to modify signature of ndot?
sol = sc.integrate.odeint(ndot, nInit, np.linspace(0, steps, steps+1))

np.savetxt("solfast.txt", sol)

# sol = np.loadtxt("solfast.txt") 

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

# fillingChart = {i: [[i-1], f'C{i}'] for i in range(2, m)}
# fillingChart[1] = [[0], 'C1']

plt.figure()
plt.fill_between(np.arange(numSamples), flippedPairListSpeciesOrder[0, :, -1], color=f'C0')
for spec in range(1, m):
    plt.fill_between(np.arange(numSamples), flippedPairListSpeciesOrder[spec-1, :, -1], flippedPairListSpeciesOrder[spec, :, -1], color=f'C{spec}')
plt.show()

