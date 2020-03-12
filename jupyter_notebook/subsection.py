import numpy
from matplotlib import pyplot
import py_monte_carlo
import scipy.sparse
import scipy.linalg

def make_ham_and_spin(nvars, edges, transverse):
    ham = numpy.zeros((2**nvars,2**nvars))
    for i in range(2**nvars):
        state = [(i >> j) & 1 for j in range(nvars)]
        h = 0.0
        for (vara, varb), j in edges:
            if state[vara] == state[varb]:
                h += j
            else:
                h -= j

        ham[i,i] += h
        for j in range(i+1, 2**nvars):
            b_state = [(j >> k) & 1 for k in range(nvars)]
            diffstate = [a^b for a,b in zip(state, b_state)]
            if sum(diffstate) != 1:
                continue
            for vark,s in enumerate(diffstate):
                if not s:
                    continue
                ham[i, j] = ham[i,j] + transverse
                ham[j, i] = ham[j,i] + transverse

    spin_diag = []
    for i in range(2**nvars):
        state = [(i >> j) & 1 for j in range(nvars)]
        spin_diag.append(numpy.sum(numpy.asarray(state)*2 - 1)**2)
    spin_op = numpy.diag(spin_diag)
    return ham, spin_op


betas = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0]

edges = [
    ((0, 1), -1.0),
    ((1, 2), 1.0),
    ((2, 3), 1.0),
    ((3, 0), 1.0),
    
    ((1, 7), 1.0),
    
    ((4, 5), -1.0),
    ((5, 6), 1.0),
    ((6, 7), 1.0),
    ((7, 4), 1.0)
]
nvars = 8
transverse = 1.0

print("ED")

large_lattice_ed_data = []
large_lattice_ed_energy = []

ham, spin_op = make_ham_and_spin(nvars, edges, transverse)

for beta in betas:
    print(beta)
    expm = scipy.linalg.expm(-beta*ham)
    large_lattice_ed_data.append(numpy.trace(spin_op @ expm) / numpy.trace(expm))
    large_lattice_ed_energy.append(numpy.trace(ham @ expm) / numpy.trace(expm))


graph = py_monte_carlo.Lattice(nvars, edges)
graph.set_transverse_field(transverse)

print("Metropolis")

large_lattice_qmc_data = numpy.zeros(len(betas))
large_lattice_qmc_energy = numpy.zeros(len(betas))

for i, beta in enumerate(betas):
    print(beta)
    results = graph.run_quantum_monte_carlo_and_measure_spins(beta, 10000, 8, exponent=2)
    results, energies = zip(*results)
    large_lattice_qmc_data[i] = numpy.mean(results, axis=0)
    large_lattice_qmc_energy[i] = numpy.mean(energies)

print("Heatbath")

large_lattice_qmc_data_heatbath = numpy.zeros(len(betas))
large_lattice_qmc_energy_heatbath = numpy.zeros(len(betas))

for i, beta in enumerate(betas):
    print(beta)
    results = graph.run_quantum_monte_carlo_and_measure_spins(beta, 10000, 8, exponent=2, use_heatbath_diagonal_update=True)
    results, energies = zip(*results)
    large_lattice_qmc_data_heatbath[i] = numpy.mean(results, axis=0)
    large_lattice_qmc_energy_heatbath[i] = numpy.mean(energies)

pyplot.plot(betas, large_lattice_ed_data, label='ED')
pyplot.plot(betas, large_lattice_qmc_data_heatbath, label='HEATBATH')
pyplot.plot(betas, large_lattice_qmc_data, label='QMC')
pyplot.xscale("log")
pyplot.legend()
pyplot.grid()
pyplot.xlabel("beta")
pyplot.ylabel("$M^2/N$")
pyplot.show()

pyplot.plot(betas, large_lattice_ed_energy, label='ED')
pyplot.plot(betas, large_lattice_qmc_energy, label='QMC')
pyplot.plot(betas, large_lattice_qmc_energy_heatbath, label='HEATBATH')
pyplot.xscale("log")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("beta")
pyplot.ylabel("E")
pyplot.show()