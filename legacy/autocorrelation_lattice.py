from bathroom_tile import graphbuilder
import py_monte_carlo
from matplotlib import pyplot


def make_lattice(j=1.0, n=8):
    gb = graphbuilder.GraphBuilder(j=j)
    gb.add_cells([
        (x, y)
        for x in range(0, n)
        for y in range(0, n)
    ])
    gb.connect_all()
    graph = gb.build(calculate_traits=False, calculate_distances=False)
    var_lookup = {v:k for k,v in enumerate(graph.all_vars)}
    edges = [((var_lookup[a], var_lookup[b]), j) for ((a,b),j) in graph.edges.items()]
    return py_monte_carlo.Lattice(edges)


def get_bond_corr(lattice, beta=39.72, transverse_field=0.1, timesteps=10000, num_experiments=1):
    lattice.set_transverse_field(transverse_field)
    return lattice.run_quantum_monte_carlo_and_measure_bond_autocorrelation(beta, timesteps, num_experiments)


if __name__ == "__main__":
    for g in [0.1, 0.25, 0.5, 0.75, 1.0]:
        lattice = make_lattice()
        corr = get_bond_corr(lattice, transverse_field=g)
        pyplot.plot(corr[0,:], label=r"$\Gamma={}$".format(g))
    pyplot.grid()
    pyplot.yscale('log')
    pyplot.legend()
    pyplot.savefig('autocorrelations.png')
    pyplot.show()