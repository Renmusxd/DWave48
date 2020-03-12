from experiments import ExperimentConfig
from dwave.system.samplers import DWaveSampler
import monte_carlo_simulator
import quantum_monte_carlo_simulator
import pickle
import os
import collections
from matplotlib import pyplot

from mocksampler import MockSampler


def make_run_dir(data_dir, run_format="run_{}"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    i = 0
    while os.path.exists(os.path.join(data_dir, run_format.format(i))):
        i += 1
    pathname = os.path.join(data_dir, run_format.format(i))
    os.mkdir(pathname)
    return pathname


def run_experiment_sweep(base_directory, experiment_gen, plot_functions=None):
    configs = None
    if not plot_functions:
        plot_functions = []
    pickle_path = os.path.join(base_directory, "configs.pickle")
    if os.path.exists(base_directory):
        if os.path.exists(pickle_path):
            print("Attempting to load experiment configurations from disk")
            with open(pickle_path, "rb") as f:
                configs = pickle.load(f)
                print("\tDone!")
    else:
        os.makedirs(base_directory)
    if configs is None:
        configs = [config for config in experiment_gen]
        print("Attempting to save experiment configurations to disk")
        with open(pickle_path, "wb") as w:
            pickle.dump(configs, w)
            print("\tDone!")

    scalars = collections.defaultdict(list)
    for i, experiment in enumerate(configs):
        experiment.run_or_load_experiment()
        experiment.add_default_analyzers()
        results = experiment.analyze()
        for k, v in results.items():
            scalars[k].append(v)
    for k, vs in scalars.items():
        with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
            for i, v in enumerate(vs):
                w.write("{}\t{}\n".format(i, v))

        pyplot.plot(vs, 'x--')
        pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
        pyplot.clf()

    for plot_fn in plot_functions:
        try:
            plot_fn(scalars)
        except Exception as e:
            print("Could not plot using {}".format(plot_fn))
            print(e)


def monte_carlo_sampler_fn():
    return monte_carlo_simulator.MonteCarloSampler()


def dwave_sampler_fn():
    return DWaveSampler()


def staggered_sampler_fn():
    return MockSampler()


def quantum_monte_carlo_fn():
    return quantum_monte_carlo_simulator.QuantumMonteCarloSampler(thermalization_time=0, timesteps=1000, sampling_freq=10)


def defect_plot(scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    defects = scalars['average_defects']
    defects_stdv = scalars['stdv_defects']
    hs = scalars['h']

    pyplot.errorbar(inv_j, defects, yerr=defects_stdv)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(experiment_name, 'defects_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.errorbar(ej_by_kt, defects, yerr=defects_stdv)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(experiment_name, 'defects_vs_ej_by_kt.svg'))
    pyplot.clf()

    pyplot.errorbar(hs, defects, yerr=defects_stdv)
    pyplot.xlabel('H')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(experiment_name, 'defects_vs_hs.svg'))
    pyplot.clf()


def flippable_plot(scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    flippable_count = scalars['flippable_count']
    flippable_stdv = scalars['flippable_stdv']
    hs = scalars['h']

    pyplot.errorbar(inv_j, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(experiment_name, 'flippable_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.errorbar(ej_by_kt, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(experiment_name, 'flippable_vs_ej_by_kt.svg'))
    pyplot.clf()

    pyplot.errorbar(hs, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('H')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(experiment_name, 'flippable_vs_hs.svg'))
    pyplot.clf()


def unit_cell_divergence_plot(scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    divergence = scalars['divergence']

    pyplot.plot(inv_j, divergence)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Boundary Divergence')
    pyplot.savefig(os.path.join(experiment_name, 'divergence_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.plot(ej_by_kt, divergence)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Boundary Divergence')
    pyplot.savefig(os.path.join(experiment_name, 'divergence_vs_ej_by_kt.svg'))
    pyplot.clf()


def experiment_generator(base_dir):
    n = 10
    for i in range(1, n+1):
        print("Building experiment {}".format(i))
        h = 0.0  # float(i) / n
        j = 1.0
        transverse = float(i) / n
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir, quantum_monte_carlo_fn,
                                  h=h, j=j, gamma=transverse, throw_errors=True)
        config.num_reads = 10
        config.auto_scale = False
        config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8)
        yield config


if __name__ == "__main__":
    experiment_name = "data/transverse/test_qmc"

    run_experiment_sweep(experiment_name, experiment_generator(experiment_name),
                         plot_functions=[defect_plot, flippable_plot, unit_cell_divergence_plot])
