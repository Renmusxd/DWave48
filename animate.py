from experiment_rewrite import BathroomTileExperiment
import py_monte_carlo
from bathroom_tile import graphbuilder
from matplotlib import pyplot


def main(beta=50.0):
    min_x = 0 + 1
    max_x = 15 - 1
    min_y = 0 + 1
    max_y = 15 - 1
    unit_cell_rect = ((min_x, max_x), (min_y, max_y))
    experiment = BathroomTileExperiment(lambda args: None,
                                        unit_cell_rect=unit_cell_rect,
                                        dwave_periodic_j=1.0, j=1.0,
                                        graph_build_kwargs={'ideal_clones': True})
    graph = experiment.graph

    var_xs = []
    var_ys = []
    for v in graph.all_vars:
        vx, vy = graphbuilder.get_var_cartesian(v)
        var_xs.append(vx)
        var_ys.append(vy)

    v_lookup = {v: k for k, v in enumerate(graph.all_vars)}

    lattice = py_monte_carlo.Lattice([((v_lookup[a], v_lookup[b]), graph.edges[(a, b)]) for (a, b) in graph.sorted_edges])
    lattice.set_initial_state([False for _ in graph.all_vars])
    samples = lattice.run_monte_carlo_sampling(beta, 10000, 1)[1][0]

    v_lookup = {v: k for k, v in enumerate(graph.all_vars)}

    last_sample = None

    aux, auy = graphbuilder.get_unit_cell_cartesian(0, 0)
    bux, buy = graphbuilder.get_unit_cell_cartesian(1, 1)
    x_spacing = abs(aux - bux) / 2.0
    y_spacing = abs(auy - buy) / 2.0

    xlim = None
    ylim = None

    for t in range(samples.shape[0]):
        if last_sample is not None and (samples[t, :] == samples[last_sample, :]).all():
            continue
        else:
            last_sample = t

        # pyplot.scatter(var_xs, var_ys, c='black')
        for a, b in graph.sorted_edges:
            if not graphbuilder.is_front(a) or not graphbuilder.is_front(b):
                continue
            a = v_lookup[a]
            b = v_lookup[b]
            pyplot.plot([var_xs[a], var_xs[b]], [var_ys[a], var_ys[b]], c='grey')

        for a, b in graph.sorted_edges:
            if not graphbuilder.is_front(a) or not graphbuilder.is_front(b):
                continue

            j = graph.edges[(a, b)]

            indx_a = v_lookup[a]
            indx_b = v_lookup[b]
            is_dimer = (samples[t, indx_a] == samples[t, indx_b]) == (j > 0)

            if is_dimer:
                a_ux, a_uy, _ = graphbuilder.get_var_traits(a)
                b_ux, b_uy, _ = graphbuilder.get_var_traits(b)

                if a_ux == b_ux and a_uy == b_uy:
                    adx, ady = graphbuilder.calculate_variable_direction(a)
                    bdx, bdy = graphbuilder.calculate_variable_direction(b)
                    dx = adx + bdx
                    dy = ady + bdy

                    ux, uy = graphbuilder.get_unit_cell_cartesian(a_ux, a_uy)
                    dux, duy = graphbuilder.get_unit_cell_cartesian(a_ux + dx, a_uy + dy)

                    pyplot.plot([ux, (ux+dux)/2.0], [uy, (uy+duy)/2.0], c='red')
                else:
                    aux, auy = graphbuilder.get_unit_cell_cartesian(a_ux, a_uy)
                    bux, buy = graphbuilder.get_unit_cell_cartesian(b_ux, b_uy)
                    avg_x = (aux + bux)/2.0
                    avg_y = (auy + buy)/2.0

                    dx = abs(a_ux - b_ux)
                    dy = abs(a_uy - b_uy)

                    pyplot.plot([avg_x - dy*x_spacing, avg_x + dy*x_spacing],
                                [avg_y - dx*y_spacing, avg_y + dx*y_spacing],
                                c='red')
        xlim = pyplot.xlim(xlim)
        ylim = pyplot.ylim(ylim)
        pyplot.savefig('state_animate/{}.png'.format(t))
        pyplot.close()

if __name__ == "__main__":
    main(5.0)