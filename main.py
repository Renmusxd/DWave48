from dwave.system.samplers import DWaveSampler
import graphbuilder
import graphanalysis
import graphdrawing
import pickle
import os
import sys
from matplotlib import pyplot
import numpy


class MockSample:
    def __getitem__(self, index):
        if index == 5:
            return 0
        if index == 13:
            return 1
        return 0


def make_run_dir(data_dir, run_format="run_{}"):
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

    response = DWaveSampler().sample_ising({}, graph, num_reads=num_reads)
    data = [({k: sample[k] for k in sample}, energy, num_occurences) for sample, energy, num_occurences in response.data()]
    with open(os.path.join(base_dir, "data.pickle"), "wb") as w:
        pickle.dump(data, w)

    return base_dir, data


def plot(base_dir, graph, data):
    def color_by_sign(var_a, var_b):
        if graph_network[(var_a, var_b)] < 0:
            return "rgb(0,0,0)"
        else:
            return "rgb(255,0,0)"

    svg = graphdrawing.make_edges_svg(list(graph), color_fn=color_by_sign)
    with open(os.path.join(base_dir, "lattice.svg"), "w") as w:
        w.write(svg)

    energies = [energy for _, energy, __ in data]
    num_occurrences = [num_occurrences for _, __, num_occurrences in data]
    pyplot.hist(energies, weights=num_occurrences)
    pyplot.savefig(os.path.join(base_dir, "energies.svg"))

    lowest_e = numpy.argmin(energies)

    sample, energy, num_occurrences = data[lowest_e]
    svg = graphdrawing.make_dimer_svg(graph, sample)
    with open(os.path.join(base_dir, "min_energy_dimers.svg"), "w") as w:
        w.write(svg)

    samples = [sample for sample, _, __ in data]
    var_map, var_mat = graphanalysis.flatten_dicts(samples)
    var_corr = graphanalysis.calculate_correlation_matrix(var_mat)

    pyplot.imshow(var_corr, interpolation='nearest')
    pyplot.savefig(os.path.join(base_dir, "correlations.svg"))


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
            for x in range(8)
            for y in range(8)
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
        plot(base_dir, graph_network, data)
