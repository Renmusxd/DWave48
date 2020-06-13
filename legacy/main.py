from experiment_gen import experiment_generator
import pickle
import collections
from bathroom_tile.experiment_metaanalysis import *
from matplotlib import pyplot
import os


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

    configs = list(enumerate(configs))

    scalars = collections.defaultdict(dict)
    while len(configs) > 0:
        i, config = configs.pop(0)
        # Load from disk if already available, otherwise recalculate things.
        config_pickle = os.path.join(config.base_dir, 'config.pickle')
        if os.path.exists(config_pickle):
            with open(config_pickle, 'rb') as f:
                experiment = pickle.load(f)
        else:
            experiment = config

        experiment.run_or_load_experiment()
        experiment.add_default_analyzers()
        results = experiment.analyze()

        with open(os.path.join(config.base_dir, 'scalars.pickle'), 'wb') as w:
            pickle.dump(results, w)
        for k, v in results.items():
            scalars[k][i] = v

        del config
        del experiment

    for k, vs in scalars.items():
        with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
            for i, v in sorted(vs.items()):
                w.write("{}\t{}\n".format(i, v))
        i_s, v_s = zip(*sorted(vs.items()))
        pyplot.plot(i_s, v_s, 'x--')
        pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
        pyplot.clf()

    with open(os.path.join(base_directory, 'scalars.pickle'), 'wb') as w:
        pickle.dump(scalars, w)

    scalar_keys = set()
    for k, v in scalars.items():
        for i in v:
            scalar_keys.add(i)

    scalar_clean = collections.defaultdict(list)
    for i in sorted(scalar_keys):
        if all(i in scalars[k] for k in scalars):
            for k in scalars:
                scalar_clean[k].append(scalars[k][i])

    for plot_fn in plot_functions:
        try:
            plot_fn(base_directory, scalar_clean)
        except Exception as e:
            print("Could not plot using {}".format(plot_fn))
            print(e)


if __name__ == "__main__":
    experiment_name = "data/dwave_full_phase_diagram"

    run_experiment_sweep(experiment_name, experiment_generator(experiment_name),
                         plot_functions=[defect_plot, flippable_plot, unit_cell_divergence_plot,
                                         flippable_phase, orientation_phase])
