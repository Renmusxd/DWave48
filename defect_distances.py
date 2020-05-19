from matplotlib import pyplot
import py_monte_carlo
from bathroom_tile import monte_carlo_simulator, graphbuilder, graphanalysis
from bathroom_tile.experiments import ExperimentConfig
from bathroom_tile.graphbuilder import get_var_cartesian, get_dimer_vertices_for_edge, get_unit_cell_cartesian, \
    get_var_traits, calculate_variable_direction, is_unit_cell_type_a
import scipy.spatial
import scipy.sparse
import numpy
import sys


def get_dimer_site_cartesian(dimer_site):
    if type(dimer_site[0]) != tuple:
        dimer_site = [dimer_site]
    dimer_sites = [get_unit_cell_cartesian(ux,uy) for ux, uy, _ in dimer_site]
    xs, ys = zip(*dimer_sites)
    return sum(xs)/float(len(dimer_site)), sum(ys)/float(len(dimer_site))


def is_periodic_link(vara, varb):
    a_unit_cell_x, a_unit_cell_y, _ = get_var_traits(vara)
    a_dx, a_dy = calculate_variable_direction(vara)
    b_unit_cell_x, b_unit_cell_y, _ = get_var_traits(varb)

    same_cell = a_unit_cell_x == b_unit_cell_x and a_unit_cell_y == b_unit_cell_y
    consistent = a_unit_cell_x + a_dx == b_unit_cell_x and a_unit_cell_y + a_dy == b_unit_cell_y

    return not same_cell and not consistent


def columnar(v):
    x, y, r = get_var_traits(v)
    if is_unit_cell_type_a(x, y):
        return (r % 2 == 0) == ((x + y) % 4 == 0)
    else:
        return (x + y) % 4 != 1


def top_right_staggered(v):
    return True


def bot_left_staggered(v):
    return v % 2 == 0


def top_left_staggered(v):
    dx, dy = calculate_variable_direction(v)

    x, y, r = get_var_traits(v)
    if y%2:
        return (dx, dy) != (0, -1)
    else:
        return (dx, dy) == (0, -1)


def get_charges(analyzer, edge_to_vertex_matrix=None):
    dimer_matrix = analyzer.get_dimer_matrix()
    diag_dimers = numpy.expand_dims(analyzer.get_diagonal_dimer_mask(), axis=-1)

    e_fields = (dimer_matrix*2 + 1) * diag_dimers

    if edge_to_vertex_matrix is None:
        edge_to_vertex_matrix = analyzer.graph.edge_to_vertex_matrix

    charges = edge_to_vertex_matrix.T @ e_fields
    charges = charges * numpy.expand_dims(numpy.asarray([(1 if len(v)==4 else -1) for v in analyzer.graph.dimer_vertex_list]), axis=-1)

    return charges

def identify_defects(graph, states, energies, state_list=None):
    variables = set()
    sorted_edges = graph.sorted_edges
    for a, b in graph.edges:
        variables.add(a)
        variables.add(b)
    variables = list(sorted(variables))
    variable_lookup = {v: k for k, v in enumerate(variables)}

    if state_list is None:
        flat_samples = (variables, states.T*2 - 1)
    else:
        flat_samples = None

    analyzer = graphanalysis.GraphAnalyzer(graph, state_list, energies, [1 for _ in states], flat_samples=flat_samples)

    charges = get_charges(analyzer)

    inner_edge_d = 1.0
    output_edge_d = 1.0

    lx, _ = get_unit_cell_cartesian(-1, 0, inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
    shift = -lx / 2.0

    points = numpy.asarray([get_var_cartesian(indx, inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
                            for indx in variables]) + shift
    edge_as = numpy.asarray([variable_lookup[a] for a, _ in sorted_edges])
    edge_bs = numpy.asarray([variable_lookup[b] for _, b in sorted_edges])
    periodic_edges = numpy.asarray([is_periodic_link(a, b) for a, b in sorted_edges])

    all_xs, all_ys = zip(*points)
    all_xs, all_ys = numpy.asarray(all_xs), numpy.asarray(all_ys)

    edge_xs = (all_xs[edge_as] + all_xs[edge_bs]) / 2.0
    edge_ys = (all_ys[edge_as] + all_ys[edge_bs]) / 2.0

    # Fix periodic edges, set to 0 along axis with variation
    # If ys are the same, set x to 0, otherwise keep x.
    x_periodic = ((all_ys[edge_as] - all_ys[edge_bs]) == 0) & periodic_edges
    y_periodic = ((all_xs[edge_as] - all_xs[edge_bs]) == 0) & periodic_edges

    edge_xs[x_periodic] = 0.0
    edge_ys[y_periodic] = 0.0

    edge_to_vertex_matrix = analyzer.graph.edge_to_vertex_matrix
    periodic_vertices = edge_to_vertex_matrix.T @ periodic_edges

    # Get the vertex cartesian positions
    # Defaults to sum of positions of edges, but special case for periodic vertices is to only consider periodic edges.
    # Removal mask removes non-periodic edges from periodic vertices
    # If a vertex is made of both x and y periodic edges, set it to zero
    simultaneous = ((edge_to_vertex_matrix.T @ x_periodic) & (edge_to_vertex_matrix.T @ y_periodic)) > 0
    simultaneous = numpy.expand_dims(simultaneous, axis=0)
    removal_mask = numpy.logical_not(
        numpy.outer(numpy.logical_not(periodic_edges), periodic_vertices > 0)) & numpy.logical_not(simultaneous)

    cleaned_edge_to_vertex = edge_to_vertex_matrix & removal_mask
    cleaned_vertex_count = numpy.sum(cleaned_edge_to_vertex, axis=0)
    cleaned_vertex_count[cleaned_vertex_count == 0] = 1

    # Now we have the vertex positions, find the ones we want to look at (with defects):
    edges_broken = analyzer.get_dimer_matrix()
    diag_dimers = numpy.expand_dims(analyzer.get_diagonal_dimer_mask(), axis=-1)

    diagonal_counts = (edge_to_vertex_matrix.T @ (edges_broken * diag_dimers == 1))
    nondiag_counts = (edge_to_vertex_matrix.T @ (edges_broken * numpy.logical_not(diag_dimers) == 1))

    # Don't consider overlapping defects
    diagonal_defects = diagonal_counts > 1
    nondiag_defects = nondiag_counts > 0

    vertex_average_x = (cleaned_edge_to_vertex.T @ edge_xs) / cleaned_vertex_count
    vertex_average_y = (cleaned_edge_to_vertex.T @ edge_ys) / cleaned_vertex_count

    extra_info = analyzer, (all_xs, all_ys), (edge_xs, edge_ys), (vertex_average_x, vertex_average_y), shift, charges

    return diagonal_defects, nondiag_defects, extra_info


def main(experiments=8, thermalization=0, timesteps=10000, n=16, beta=3.0):
    config = ExperimentConfig('/tmp/experiment',
                              monte_carlo_simulator.MonteCarloSampler,
                              build_kwargs={'ideal_periodic_boundaries': True})
    config.build_graph(min_x=0, max_x=n, min_y=0, max_y=n, calculate_distances=False)
    edges = config.graph.edges
    variables = set()
    for a, b in edges:
        variables.add(a)
        variables.add(b)
    variables = list(sorted(variables))
    variable_lookup = {v: k for k, v in enumerate(variables)}

    lattice_edges = [((variable_lookup[a], variable_lookup[b]), j) for (a, b), j in edges.items()]
    lattice = py_monte_carlo.Lattice(lattice_edges)

    lattice.set_initial_state([columnar(v) for v in variables])

    print("Running Monte Carlo")
    energies, states = lattice.run_monte_carlo_sampling(beta, timesteps, experiments,
                                                        thermalization_time=thermalization)
    states = states.reshape((-1, states.shape[-1]))

    print("Identifying defects")
    _, _, extra_info = identify_defects(config.graph, states, energies)
    analyzer, _, _, (vertex_average_x, vertex_average_y), shift, charges = extra_info

    inner_edge_d = 1.0
    output_edge_d = 1.0

    lx, _ = get_unit_cell_cartesian(-1, 0, inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
    shift = -lx / 2.0

    # Get the periodicity
    wrap_x, _ = get_unit_cell_cartesian(analyzer.graph.used_unit_cells_per_row, 0, inner_edge_d=inner_edge_d,
                                        output_edge_d=output_edge_d)
    wrap_x += shift

    x_shift = numpy.zeros((1, 2))
    x_shift[0, 0] = wrap_x
    y_shift = numpy.zeros((1, 2))
    y_shift[0, 1] = wrap_x

    def distances_on_torus(a_coords, b_coords):
        spaces = []
        for dx in [0, -1, 1]:
            for dy in [0, -1, 1]:
                spaces.append(scipy.spatial.distance_matrix(a_coords + x_shift * dx + y_shift * dy, b_coords))
        spaces = numpy.asarray(spaces)
        return numpy.amin(spaces, axis=0)

    all_vertices = numpy.asarray([vertex_average_x, vertex_average_y]).T
    all_distances = distances_on_torus(all_vertices, all_vertices).flatten()

    all_prong_defects = charges > 0
    all_bar_defects = charges < 0

    print("Calculating distances")
    dists = []
    for t in range(charges.shape[-1]):
        prong_defects = all_prong_defects[:,t]
        bar_defects = all_bar_defects[:,t]

        if any(prong_defects) and any(bar_defects):
            prong_xs = vertex_average_x[prong_defects]
            prong_ys = vertex_average_y[prong_defects]
            bar_xs = vertex_average_x[bar_defects]
            bar_ys = vertex_average_y[bar_defects]
            prongs = numpy.dstack((prong_xs, prong_ys))[0]
            bars = numpy.dstack((bar_xs, bar_ys))[0]

            # Remember, we are on a torus, space is weird.
            min_distances = numpy.amin(distances_on_torus(prongs, bars), axis=-1)
            dists.extend(min_distances)

    pyplot.hist(dists, density=True)
    pyplot.hist(all_distances, histtype='step', density=True)
    if len(sys.argv) > 1:
        pyplot.savefig(sys.argv[1])
    pyplot.show()


if __name__ == "__main__":
    experiments = 1
    timesteps = 100000
    n = 16
    main(experiments=experiments, timesteps=timesteps, n=n, beta=2.0)

