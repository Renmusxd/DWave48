from matplotlib import pyplot
import py_monte_carlo
from bathroom_tile import monte_carlo_simulator, graphbuilder, graphanalysis
from bathroom_tile.experiments import ExperimentConfig
from bathroom_tile.graphbuilder import get_var_cartesian, get_dimer_vertices_for_edge, get_unit_cell_cartesian, \
    get_var_traits, calculate_variable_direction, is_unit_cell_type_a
from bathroom_tile.graphdrawing import get_var_pos
import numpy

from defect_distances import identify_defects


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


if __name__ == "__main__":
    config = ExperimentConfig('/tmp/experiment',
                              monte_carlo_simulator.MonteCarloSampler,
                              build_kwargs={'ideal_periodic_boundaries': True})
    config.build_graph(min_x=0, max_x=16, min_y=0, max_y=16, calculate_traits=True, calculate_distances=False)
    edges = config.graph.edges
    variables = set()
    for a, b in edges:
        variables.add(a)
        variables.add(b)
    variables = list(sorted(variables))
    variable_lookup = {v: k for k, v in enumerate(variables)}

    lattice_edges = [((variable_lookup[a], variable_lookup[b]), j) for (a, b), j in edges.items()]
    lattice = py_monte_carlo.Lattice(lattice_edges)

    lattice.set_initial_state([top_right_staggered(v) for v in variables])

    timesteps = 10000
    energies, states = lattice.run_monte_carlo_sampling(3.0, timesteps, 1)
    energies = energies[0]
    states = states[0]

    _, _, extra_info = identify_defects(config.graph, states, energies)
    _, _, _, (vertex_average_x, vertex_average_y), shift, charges = extra_info

    positive_defects = charges > 0
    negative_defects = charges < 0

    vertex_average_x -= shift
    vertex_average_y -= shift

    points = [get_var_cartesian(indx) for indx in variables]
    all_xs, all_ys = zip(*points)
    last_hash = 0
    # For consistency
    xlim, ylim = None, None
    for i, (e, state) in enumerate(zip(energies, states)):
        new_hash = hash(tuple(state))
        if new_hash != last_hash:
            last_hash = new_hash

            for a, b in edges:
                j = edges[(a, b)]
                sa = state[variable_lookup[a]]
                sb = state[variable_lookup[b]]

                if (sa == sb and j > 0.0) or (sa != sb and j < 0.0):

                    if not is_periodic_link(a, b):
                        d_site_alpha, d_site_beta = get_dimer_vertices_for_edge(a, b)
                        alpha_x, alpha_y = get_dimer_site_cartesian(d_site_alpha)
                        beta_x, beta_y = get_dimer_site_cartesian(d_site_beta)
                        pyplot.plot([alpha_x, beta_x], [alpha_y, beta_y], c='r')
                    else:
                        for v in [a, b]:
                            vux, vuy, _ = get_var_traits(v)
                            dx, dy = calculate_variable_direction(v)

                            d_site_alpha = tuple(sorted([
                                (vux, vuy, True),
                                (vux+dx, vuy+dy, True),
                                (vux + dy, vuy + dx, True),
                                (vux + dx + dy, vuy + dy + dx, True),
                            ]))

                            d_site_beta = tuple(sorted([
                                (vux, vuy, True),
                                (vux + dx, vuy + dy, True),
                                (vux - dy, vuy - dx, True),
                                (vux + dx - dy, vuy + dy - dx, True),
                            ]))

                            alpha_x, alpha_y = get_dimer_site_cartesian(d_site_alpha)
                            beta_x, beta_y = get_dimer_site_cartesian(d_site_beta)
                            pyplot.plot([alpha_x, beta_x], [alpha_y, beta_y], c='r')

            on_point_xs = [x for v, x in enumerate(all_xs) if state[v]]
            on_point_ys = [y for v, y in enumerate(all_ys) if state[v]]
            off_point_xs = [x for v, x in enumerate(all_xs) if not state[v]]
            off_point_ys = [y for v, y in enumerate(all_ys) if not state[v]]

            pyplot.title("E: {}".format(e))

            pos_xs, pos_ys = vertex_average_x[positive_defects[:,i]], vertex_average_y[positive_defects[:,i]]
            neg_xs, neg_ys = vertex_average_x[negative_defects[:,i]], vertex_average_y[negative_defects[:,i]]
            pyplot.scatter(pos_xs, pos_ys, marker='o', c='b')
            pyplot.scatter(neg_xs, neg_ys, marker='x', c='r')

            pyplot.scatter(on_point_xs, on_point_ys, 0.1, marker='o', c='g')
            pyplot.scatter(off_point_xs, off_point_ys, 0.1, marker='o', c='b')
            xlim = pyplot.xlim(xlim)
            ylim = pyplot.ylim(ylim)
            pyplot.gca().invert_yaxis()
            pyplot.savefig('state_animate/{}.png'.format(str(i).zfill(len(str(timesteps)))))
            pyplot.clf()