from dwave.system.samplers import DWaveSampler
import graphbuilder
import graphanalysis
import graphdrawing
import pickle
import os
import sys
from matplotlib import pyplot
import numpy


def make_run_dir(data_dir, run_format="run_{}"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    i = 0
    while os.path.exists(os.path.join(data_dir, run_format.format(i))):
        i += 1
    pathname = os.path.join(data_dir, run_format.format(i))
    os.mkdir(pathname)
    return pathname


def make_run_and_save(graph, hs=None, num_reads=0, data_dir="./data", run_format="run_{}", auto_scale=True):
    if hs is None:
        hs = {}
    base_dir = make_run_dir(data_dir, run_format=run_format)

    with open(os.path.join(base_dir, "network.pickle"), "wb") as w:
        pickle.dump(graph, w)

    if num_reads:
        response = DWaveSampler().sample_ising(hs, graph, num_reads=num_reads, auto_scale=auto_scale)
        data = [({k: sample[k] for k in sample}, energy, num_occurences) for sample, energy, num_occurences in
                response.data()]
        with open(os.path.join(base_dir, "data.pickle"), "wb") as w:
            pickle.dump(data, w)
    else:
        data = None

    return base_dir, data


def draw_dimers(filename, graph, sample, front=True, color_on_orientation=True):

    def color_on_orientation_fn(var_a, var_b):
        dx_a, dy_a = graphanalysis.calculate_variable_direction(var_a)
        dx_b, dy_b = graphanalysis.calculate_variable_direction(var_b)
        # Vertical and horizontal bonds are green (and rare)
        if dx_a == -dx_b or dy_a == -dy_b:
            return "green"
        if dx_a == dy_b and dy_a == dx_b:
            return "red"
        if dx_a == -dy_b and dy_a == -dx_b:
            return "blue"
        return "purple"

    if not color_on_orientation:
        # Basic color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front)
        if svg:
            with open(filename, "w") as w:
                w.write(svg)
    else:
        # Orientation color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front, dimer_color_fn=color_on_orientation_fn)
        if svg:
            with open(filename, "w") as w:
                w.write(svg)


def draw_occupations(filename, graph, graph_analyzer, front=True):
    edge_lookup, dimer_matrix = graph_analyzer.get_dimer_matrix()
    average_dimers = numpy.mean(dimer_matrix == 1, -1)
    stdv_dimers = numpy.var(dimer_matrix == 1, -1)

    # Basic color scheme
    def dimer_color_fn(var_a, var_b):
        edge = (min(var_a, var_b), max(var_a, var_b))
        edge_indx = edge_lookup[edge]
        average_values = average_dimers[edge_indx]
        red_color = int(average_values*256)
        return "rgb({},0,0)".format(red_color)

    svg = graphdrawing.make_dimer_contents(graph, front=front, dimer_color_fn=dimer_color_fn)
    svg = graphdrawing.wrap_with_svg(svg)
    if svg:
        with open(filename, "w") as w:
            w.write(svg)

    return average_dimers, stdv_dimers

def plot(base_dir, graph, data=None):
    def color_by_sign(var_a, var_b):
        if graph_network[(var_a, var_b)] < 0:
            return "rgb(0,0,0)"
        else:
            return "rgb(255,0,0)"

    svg = graphdrawing.make_edges_svg(list(graph), color_fn=color_by_sign, front=True)
    if svg:
        with open(os.path.join(base_dir, "front_lattice.svg"), "w") as w:
            w.write(svg)
    svg = graphdrawing.make_edges_svg(list(graph), color_fn=color_by_sign, front=False)
    if svg:
        with open(os.path.join(base_dir, "rear_lattice.svg"), "w") as w:
            w.write(svg)

    if data:
        # # First plot the energies that we found
        # energies = [energy for _, energy, __ in data]
        # num_occurrences = [num_occurrences for _, __, num_occurrences in data]
        # pyplot.hist(energies, weights=num_occurrences)
        # pyplot.savefig(os.path.join(base_dir, "energies.svg"))
        #
        # # Then plot the dimers for one of the ground states (or lowest E we found anyway).
        # lowest_e = numpy.argmin(energies)
        # sample, energy, num_occurrences = data[lowest_e]
        # draw_dimers(os.path.join(base_dir, "front_min_energy_dimers.svg"), graph, sample,
        #             front=True, color_on_orientation=False)
        # draw_dimers(os.path.join(base_dir, "front_min_energy_dimers_color.svg"), graph, sample,
        #             front=True, color_on_orientation=True)
        #
        # draw_dimers(os.path.join(base_dir, "rear_min_energy_dimers.svg"), graph, sample,
        #             front=False, color_on_orientation=False)
        # draw_dimers(os.path.join(base_dir, "rear_min_energy_dimers_color.svg"), graph, sample,
        #             front=False, color_on_orientation=False)
        #
        # # Now plot the correlation matrix for the variables, and the correlation as a function of distance.
        samples = [sample for sample, _, __ in data]
        #
        graph_analyzer = graphanalysis.GraphAnalyzer(graph, samples)
        #
        # var_corr, distance_corr, distance_stdv, all_vars = graph_analyzer.calculate_correlation_function()
        # distance_corr = numpy.nan_to_num(distance_corr)
        # distance_stdv = numpy.nan_to_num(distance_stdv)
        # pyplot.imshow(var_corr, interpolation='nearest')
        # pyplot.colorbar()
        # pyplot.savefig(os.path.join(base_dir, "variable_correlations.svg"))
        # pyplot.clf()
        #
        # # Now the distance part, with error bars.
        # average_corrs = numpy.mean(distance_corr, 0)
        # stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
        # xs = numpy.arange(average_corrs.shape[0])
        # pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
        # pyplot.legend()
        # pyplot.grid()
        # pyplot.xlabel("Distance (in # edges)")
        # pyplot.ylabel("Correlation")
        # pyplot.savefig(os.path.join(base_dir, "correlation_distance.svg"))
        # pyplot.clf()
        #
        # # Similarly, get the correlation plot for the dimers
        # _, dimer_corrs = graph_analyzer.get_dimer_correlations()
        # pyplot.imshow(dimer_corrs, interpolation='nearest')
        # pyplot.colorbar()
        # pyplot.savefig(os.path.join(base_dir, "dimer_correlations.svg"))
        # pyplot.clf()
        #
        # # Get the average dimer occupations
        # average_dimers, stdv_dimers = draw_occupations(os.path.join(base_dir, "dimer_occputation_graph.svg"),
        #                                                graph, graph_analyzer)
        # # Sort them
        # average_dimers, stdv_dimers = zip(*sorted(zip(average_dimers, stdv_dimers), key=lambda x: x[0]))
        # average_dimers = numpy.asarray(average_dimers)
        # stdv_dimers = numpy.asarray(stdv_dimers)
        # xs = numpy.arange(len(average_dimers))
        # pyplot.plot(xs, average_dimers)
        # pyplot.plot(xs, average_dimers+stdv_dimers, 'r--')
        # pyplot.plot(xs, average_dimers-stdv_dimers, 'r--')
        # pyplot.savefig(os.path.join(base_dir, "dimer_occupations_plot.svg"))
        # pyplot.clf()

        defects = graph_analyzer.get_defects()


def data_loader_generator(dirs):
    for base_dir in dirs:
        with open(os.path.join(base_dir, "network.pickle"), "rb") as f:
            graph_network = pickle.load(f)
        with open(os.path.join(base_dir, "data.pickle"), "rb") as f:
            data = pickle.load(f)
        yield base_dir, graph_network, data


if __name__ == "__main__":
    run_args = []
    if len(sys.argv) == 1:
        graph = graphbuilder.Graph(j=1.0)

        Lx = 8
        Ly = 16
        graph.add_cells([
            (x, y)
            for x in range(0, Lx)
            for y in range(0, Ly)
        ])
        graph.connect_all()
        hs, graph_network = graph.build()
        hs.update({
            # 925: 5.0,
            # 933: -5.0,
        })

        base_dir, data = make_run_and_save(graph_network, data_dir="data", hs=hs, num_reads=10000,
                                           auto_scale=True)
        run_args = [(base_dir, graph_network, data)]
    elif len(sys.argv) > 1:
        # Replace with a generator so that it doesn't load too much at once
        run_args = data_loader_generator(sys.argv[1:])

    # Go through each thing to run
    for base_dir, graph_network, data in run_args:
        plot(base_dir, graph_network, data=data)
