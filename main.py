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
        os.mkdir(data_dir)
    i = 0
    while os.path.exists(os.path.join(data_dir, run_format.format(i))):
        i += 1
    pathname = os.path.join(data_dir, run_format.format(i))
    os.mkdir(pathname)
    return pathname


def make_run_and_save(graph, num_reads=1000, data_dir="./data", run_format="run_{}"):
    base_dir = make_run_dir(data_dir, run_format=run_format)

    with open(os.path.join(base_dir, "network.pickle"), "wb") as w:
        pickle.dump(graph, w)

    if num_reads:
        response = DWaveSampler().sample_ising({}, graph, num_reads=num_reads)
        data = [({k: sample[k] for k in sample}, energy, num_occurences) for sample, energy, num_occurences in
                response.data()]
        with open(os.path.join(base_dir, "data.pickle"), "wb") as w:
            pickle.dump(data, w)
    else:
        data = None

    return base_dir, data


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
        energies = [energy for _, energy, __ in data]
        num_occurrences = [num_occurrences for _, __, num_occurrences in data]
        pyplot.hist(energies, weights=num_occurrences)
        pyplot.savefig(os.path.join(base_dir, "energies.svg"))

        lowest_e = numpy.argmin(energies)

        sample, energy, num_occurrences = data[lowest_e]
        front_svg = graphdrawing.make_dimer_svg(graph, sample, front=True)
        if front_svg:
            with open(os.path.join(base_dir, "front_min_energy_dimers.svg"), "w") as w:
                w.write(front_svg)
        rear_svg = graphdrawing.make_dimer_svg(graph, sample, front=False)
        if rear_svg:
            with open(os.path.join(base_dir, "rear_min_energy_dimers.svg"), "w") as w:
                w.write(rear_svg)
        if front_svg and rear_svg:
            svg = graphdrawing.make_combined_dimer_svg(graph, sample)
            if svg:
                with open(os.path.join(base_dir, "min_energy_dimers.svg"), "w") as w:
                    w.write(svg)

        samples = [sample for sample, _, __ in data]
        var_corr, distance_corr, distance_stdv, all_vars = graphanalysis.calculate_correlation_function(graph, samples)
        distance_corr = numpy.nan_to_num(distance_corr)
        distance_stdv = numpy.nan_to_num(distance_stdv)

        pyplot.imshow(var_corr, interpolation='nearest')
        pyplot.savefig(os.path.join(base_dir, "correlations.svg"))
        pyplot.clf()

        average_corrs = numpy.mean(distance_corr, 0)
        stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))

        middle_index = len(all_vars) // 2
        xs = numpy.arange(average_corrs.shape[0])
        pyplot.errorbar(xs, distance_corr[middle_index, :], yerr=distance_stdv[middle_index, :],
                        label="Variable: {}".format(all_vars[middle_index]))
        pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
        pyplot.legend()
        pyplot.grid()
        pyplot.xlabel("Distance (in # edges)")
        pyplot.ylabel("Correlation")
        pyplot.savefig(os.path.join(base_dir, "correlation_distance.svg"))
        pyplot.clf()


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
        graph = graphbuilder.Graph()
        graph.add_cells([
            (x, y)
            for x in range(16)
            for y in range(16)
        ])
        graph.connect_all()
        graph_network = graph.build()
        base_dir, data = make_run_and_save(graph_network)
        run_args = [(base_dir, graph_network, data)]
    elif len(sys.argv) > 1:
        # Replace with a generator so that it doesn't load too much at once
        run_args = data_loader_generator(sys.argv[1:])

    # Go through each thing to run
    for base_dir, graph_network, data in run_args:
        plot(base_dir, graph_network, data=data)
