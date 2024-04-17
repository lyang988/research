import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from copy import deepcopy

supply = 1 # total nutrient supply
delta = supply 
e = 1 # total enzyme budget

timebound = 760 # paper uses 1200
mu = 0.05 # Parameter for migration rate; maybe change to exponential
timestep = 1

d = 1 # diffusion coefficient
l = 20 # length of the system
s = np.array([.25, .45, .3]) # nutrient supply
p = len(s) # number of nutrients
np.random.seed(1805)

original_m = 100 # number of species
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
    return pops, False
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
    

# Removes segments with population size less than lambda, redistributing their population to their neighbors
# - half each
def removeSmallSegments(pops, lamb):
    global positionToSpecies

    segs_to_delete = pops < lamb
    new_pops = np.zeros(len(pops) - sum(segs_to_delete))

    i = 0
    j = 0
    while i < len(pops):
        if segs_to_delete[i]:
            if j > 0 and j < len(new_pops):
                new_pops[j-1] += pops[i] / 2
                new_pops[j] += pops[i] / 2
            elif j == 0:
                new_pops[j] += pops[i]
            else:
                # j == len(new_pops)
                new_pops[j-1] += pops[i]

            i += 1
            continue

        new_pops[j] += pops[i]
        j += 1
        i += 1

    # Update positionToSpecies
    new_pos_to_species = [spec for i, spec in enumerate(positionToSpecies) if not segs_to_delete[i]]

    # Iterate through positionToSpecies, "merging" species that are the same
    segs_to_merge = [i != 0 and new_pos_to_species[i] == new_pos_to_species[i-1] for i in range(len(new_pos_to_species))]
    merged_pops = np.zeros(len(new_pops) - sum(segs_to_merge))

    i = 0
    j = 0
    while i < len(new_pops):
        if segs_to_merge[i]:
            merged_pops[j-1] += new_pops[i]
        else:
            merged_pops[j] += new_pops[i]
            j += 1
        i += 1
    
    positionToSpecies = [spec for i, spec in enumerate(new_pos_to_species) if not segs_to_merge[i]]

    return merged_pops

# tests the above function:
# print(removeSmallSegments(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.2))
# print(removeSmallSegments(np.array([0.5, 0.1, 0.3, 0.4, 0.5]), 0.2))
# print(removeSmallSegments(np.array([0.5, 0.1, 0.3, 0.4, 0.1]), 0.2))
# exit()

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
    nToEliminate, succeeded = performMigration(nToMigrate)
    if succeeded:
        successful_migrations += 1
    else:
        failed_migrations += 1
    
    # Remove segments with population size less than lambda
    # Lambda is l / num_species / 1000
    nCur = removeSmallSegments(nToEliminate, l / original_m / 10)

print(f"Successful migrations: {successful_migrations}; Failed migrations: {failed_migrations}")

# for i, sol in enumerate(collectedSols):
#     np.savetxt(f"migration/solmigrationmore{i}.txt", sol)

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

# Output total distance each species covers, number of species with population size greater than lambda

# Sol is an array/list of length m
def analyze(sol, lamb):
    m = len(sol)
    species_cover = [0] * original_m
    for i in range(m):
        species_cover[positionToSpecies[i]] += sol[i]

    # Num segments (not species) with population size greater than lamb
    num_seg_large = sum(1 for x in sol if x > lamb)
    num_spec_large = sum(1 for x in species_cover if x > lamb)

    return species_cover, num_seg_large, num_spec_large

# Create 3x2 plot
fig, axs = plt.subplots(3, 2)
for i, (sol, times, posToSpecies) in enumerate(collectedSols):
    m = len(posToSpecies)
    sampleTimes = times.astype(int)
    numSamples = len(sampleTimes)

    popSamples = np.array([sol[t] for t in range(numSamples)])

    positionSamples = np.array([getPositions(pop) for pop in popSamples])

    # TODO: check this function
    boundarySamples = np.array([[positionSamples[i, j, 1] for j in range(m)] for i in range(numSamples)])

    pairListTimeOrder = np.array([np.transpose([boundarySamples[i], np.ones(m)*sampleTimes[i]]) for i in range(numSamples)])

    pairListSpeciesOrder = np.transpose(pairListTimeOrder, (1, 0, 2))

    flippedPairListSpeciesOrder = np.array([[pairListSpeciesOrder[i, j][::-1] for j in range(numSamples)] for i in range(m)])

    # fillingChart = {i: [[i-1], f'C{i}'] for i in range(2, m)}
    # fillingChart[1] = [[0], 'C1']

    axs[0][0].fill_between(times, flippedPairListSpeciesOrder[0, :, -1], color=f'C{posToSpecies[0]}')
    for spec in range(1, m):
        axs[0][0].fill_between(times, flippedPairListSpeciesOrder[spec-1, :, -1], flippedPairListSpeciesOrder[spec, :, -1], color=f'C{posToSpecies[spec]}')

axs[0][0].set_xlabel("Time")
axs[0][0].set_ylabel("Position")

final_sol = collectedSols[-1][0][-1]
final_m = len(final_sol)
final_sol_postospec = collectedSols[-1][2]

lamb = 0.1

species_cover, num_seg_large, num_spec_large = analyze(final_sol, lamb)
# Plot the final_sol as a stacked bar
for i in range(len(species_cover)):
    axs[0][1].bar([0], [species_cover[i]], bottom=[0 if i == 0 else sum(species_cover[:i])], color=[f"C{i}"])
axs[0][1].set_xticks([])
axs[0][1].set_xlabel("Species distribution")
axs[0][1].set_ylabel("Population Size")

# Sort the species covers
sorted_species_cover = sorted(enumerate(species_cover), key=lambda v: v[1], reverse=True)
# Plot as bar chart
axs[1][0].bar(np.arange(len(sorted_species_cover)), [v[1] for v in sorted_species_cover], color=[f"C{i}" for i, _ in sorted_species_cover])
axs[1][0].set_xticks(np.arange(len(sorted_species_cover)))
axs[1][0].set_xticklabels([f"{i}" for i, _ in sorted_species_cover])
axs[1][0].set_xlabel("Species")
axs[1][0].set_ylabel("Population Size")

# Plot histogram of species covers
hist, bins = np.histogram(final_sol, bins=20)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
axs[1][1].hist(final_sol, bins=logbins)
axs[1][1].set_xscale('log')
axs[1][1].set_xlabel("Population Size")
axs[1][1].set_ylabel("Density")

print(f"Number of migrations: {len(collectedSols) - 1}; Number of segments with population size greater than {lamb}: {num_seg_large}; number of species as such: {num_spec_large}; number of nonzero species: {sum(1 for x in species_cover if x > 0)}")

def plot_resource_graph(sol, sol_postospec):
    sol_cumsum = np.insert(np.cumsum(sol), 0, 0)
    sol_m = len(sol)
    # Plots final resource graph
    for nutrient in range(p):
        ABCoeff = getConcentration(sol, nutrient, sol_postospec)
        poses = np.linspace(0, l, 3 * sol_m, endpoint=False)
        concs = []
        for pos in poses:
            segment = np.searchsorted(sol_cumsum, pos, side='right') - 1
            species = sol_postospec[segment]
            conc = calculateConcentrationForNutrientAtPoint(pos - sol_cumsum[segment], species, nutrient, ABCoeff[segment], ABCoeff[segment + sol_m])
            concs.append(conc)
        
        axs[2][0].scatter(poses, concs, color=f"C{nutrient}")

# plot_resource_graph(collectedSols[0][0][0], collectedSols[0][2])
plot_resource_graph(final_sol, final_sol_postospec)

plt.show()
