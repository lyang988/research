import numpy as np
import scipy as sc
from copy import deepcopy
import sys

supply = 1 # total nutrient supply
delta = supply 
e = 1 # total enzyme budget

timebound = 6 # paper uses 1200
mu = 0.5 # Parameter for migration rate; maybe change to exponential
timestep = 1

d = 1 # diffusion coefficient
l = 20 # length of the system
s = np.array([.25, .45, .3]) # nutrient supply
p = len(s) # number of nutrients
np.random.seed(28)

original_m = 10 # number of species
migration_width = l / (5 * original_m) # width of migration zone

def randomStrats(number):
    # ? Make this explicitly pulling from simplex if doesn't work
    return np.random.dirichlet(np.ones(p), number)
alpha = randomStrats(original_m)

positionToSpecies = list(range(original_m))


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

def getConcentration(n, nutrient, posToSpec): 
    m0 = len(n)
    # print(f"m0 is {m0}")
    abVarMatrix = np.zeros((2 * m0, 2 * m0))
    finalMatrix = np.zeros((2 * m0,))
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
        species = posToSpec[i]
        nextSpecies = posToSpec[nextIndex]

        # First half of the requirements on A and B: c(\[Theta]) is continuous at the boundaries
        abVarMatrix[i, i] = cA(alpha[species, nutrient], n[i])
        abVarMatrix[i, i + bVarOffset] = cB(alpha[species, nutrient], n[i])
        abVarMatrix[i, nextIndex] = -cA(alpha[nextSpecies, nutrient], 0)
        abVarMatrix[i, nextIndex + bVarOffset] = -cB(alpha[nextSpecies, nutrient], 0)
        finalMatrix[i] -= cC(alpha[species, nutrient])
        finalMatrix[i] += cC(alpha[nextSpecies, nutrient]) # These are 'flipped' because we 'move' the terms to the other side of the equation

        # Second half of the requirements on A and B: the derivatives of c(\[Theta]) are continuous at the boundaries
        abVarMatrix[i + m0, i] = dcA(alpha[species, nutrient], n[i])
        abVarMatrix[i + m0, i + bVarOffset] = dcB(alpha[species, nutrient], n[i])
        abVarMatrix[i + m0, nextIndex] = -dcA(alpha[nextSpecies, nutrient], 0)
        abVarMatrix[i + m0, nextIndex + bVarOffset] = -dcB(alpha[nextSpecies, nutrient], 0)
        finalMatrix[i + m0] -= dcC()
        finalMatrix[i + m0] += dcC() # These are 'flipped' because we 'move' the terms to the other side of the equation

    solution = np.linalg.solve(abVarMatrix, finalMatrix)
    # Solution is a 2m0 array, solution[i] = A_i, solution[i + m0] = B_i
    # Note that this is only for one nutrient
    return solution

def calculateConcentrationForNutrientAtPoint(positionWithinSegment, speciesAtPosition, nutrient, A, B):
    # A and B must be the relevant A and B values for the nutrient at this position
    firstTerm = s[nutrient] / alpha[speciesAtPosition, nutrient]
    secondTerm = A * np.exp(positionWithinSegment * np.sqrt(alpha[speciesAtPosition, nutrient] / d))
    thirdTerm = B * np.exp(-positionWithinSegment * np.sqrt(alpha[speciesAtPosition, nutrient] / d))
    return firstTerm + secondTerm + thirdTerm

# Population Dynamics Functions: 

def getGrowth(m0, pop, pos, c):
    species = positionToSpecies[pos]
    myN = pop[pos]
    myA = c[pos] # 0/1 indexing? Should this be species*2? Similarly for the next: + 1?
    myB = c[m0 + pos]
    growth = np.array([alpha[species, i]*((s[i]*myN/alpha[species, i]) + np.sqrt(d/alpha[species, i])*(myA[i]*(np.exp(myN*np.sqrt(alpha[species, i]/d)) - 1) - myB[i]*(np.exp(-myN*np.sqrt(alpha[species, i]/d)) - 1))) for i in range(p)]).sum()
    return growth


def ndot(n, t=0):
    # You can use t if you want to make the growth rate time-dependent
    n0 = n
    m0 = len(n0)
    cCoeff = np.array([getConcentration(n0, v, positionToSpecies) for v in range(p)]).T
    growthTable = np.array([getGrowth(m0, n0, i, cCoeff) for i in range(m0)])
    dn = (growthTable - delta*n0).astype(np.float64)
    return dn

def performMigration(pops):
    global positionToSpecies
    pop_cumsum = np.insert(np.cumsum(pops), 0, 0)
    num_segs = len(pops)
    for _ in range(100):
        # This is the migration step
        pos_mig_from = np.random.uniform(0, l)
        pos_mig_to = np.random.uniform(0, l)

        # If they overlap, try again
        if abs(pos_mig_from - pos_mig_to) < migration_width:
            continue
        
        # If either overlaps with any population boundary, try again
        if any(pos_mig_from < boundary < pos_mig_from + migration_width for boundary in pop_cumsum):
            continue
        if any(pos_mig_to < boundary < pos_mig_to + migration_width for boundary in pop_cumsum):
            continue

        # Find the populations that are being migrated
        # Finds index of last element less than pos_mig_from
        from_pop = np.searchsorted(pop_cumsum, pos_mig_from, side='right') - 1
        to_pop = np.searchsorted(pop_cumsum, pos_mig_to, side='right') - 1

        # If these are the same, try again
        if positionToSpecies[from_pop] == positionToSpecies[to_pop]:
            continue

        # If we get here, we can perform the migration
        break
    else:
        print("Failed to find migration")
        return pops, False

    # TODO: check off-by-ones

    print(f"Performing migration from {pos_mig_from} to {pos_mig_to}")

    # Perform the competition between old and new populations
    # Based on dot product of concentrations
    
    # A and B array
    cCoeff = np.array([getConcentration(pops, v, positionToSpecies) for v in range(p)]).T
    nutrient_concentrations = np.zeros((p,))
    for nutrient in range(p):
        A, B = cCoeff[to_pop, nutrient], cCoeff[to_pop + num_segs, nutrient]
        # Here is where we take the 'middle' concentration
        conc = calculateConcentrationForNutrientAtPoint(pos_mig_to - pop_cumsum[to_pop] + migration_width / 2, positionToSpecies[to_pop], nutrient, A, B)
        nutrient_concentrations[nutrient] = conc
    
    # Get dot product of nutrient_concentrations and alpha[to_pop] and alpha[from_pop]
    # If the dot product for the migrant is greater, the migrant wins
    migrant_score = np.dot(nutrient_concentrations, alpha[positionToSpecies[from_pop]])
    non_migrant_score = np.dot(nutrient_concentrations, alpha[positionToSpecies[to_pop]])
    if migrant_score < non_migrant_score:
        print("Migrant loses")
        return pops, False
    
    # For now, we leave the migrating population as-is
    populations_unaffected_before = pops[:to_pop]
    populations_unaffected_after = pops[to_pop+1:]

    affected_pops = np.array([pos_mig_to - pop_cumsum[to_pop], migration_width, pop_cumsum[to_pop+1] - pos_mig_to - migration_width])

    # Modifies \alpha to account for this migration
    # alpha = np.insert(alpha, to_pop + 1, alpha[from_pop], axis=0)
    # alpha = np.insert(alpha, to_pop + 2, alpha[to_pop], axis=0)
    
    # Modifies position_to_species to account for this migration
    positionToSpecies = positionToSpecies[:to_pop + 1] + [positionToSpecies[from_pop]] + positionToSpecies[to_pop:]

    return np.concatenate((populations_unaffected_before, affected_pops, populations_unaffected_after)), True
    

# Initial Conditions and Integration

nInit = l*np.ones(original_m)/original_m # initial population

collectedSols = []

nCur = np.array(nInit)
t = 0
successful_migrations = 0
failed_migrations = 0
while t < timebound:
    print(f"Working: {t}")
    time_to_next_migration = np.random.geometric(mu)
    new_t = min(t + time_to_next_migration, timebound)
    t_delta = new_t - t
    times = np.linspace(t, new_t, 1 + int(t_delta/timestep))
    stepSol = sc.integrate.odeint(ndot, nCur, times)
    t = new_t

    # if collectedSols is None:
    #     # We don't add the last one, as we need to perfom migration!
    #     collectedSols = stepSol[:-1].copy()
    # else:
    #     collectedSols = np.append(collectedSols, stepSol[:-1], axis=0)
    collectedSols.append((stepSol.copy(), times.copy(), deepcopy(positionToSpecies)))

    print(positionToSpecies)

    nToMigrate = stepSol[-1]

    # This is the migration step
    nCur, succeeded = performMigration(nToMigrate)
    if succeeded:
        successful_migrations += 1
    else:
        failed_migrations += 1

print(f"Successful migrations: {successful_migrations}; Failed migrations: {failed_migrations}")

for i, sol in enumerate(collectedSols):
    np.savetxt(f"migration/{sys.argv[1]}-sol{i}.txt", sol)
