from bathroom_tile.experiment_metaanalysis import flippable_phase, orientation_phase
from experiment_gen import experiment_generator
import experiment_rewrite
import bathroom_tile
from matplotlib import pyplot
import numpy
import argparse
import os
import pickle
import collections
import multiprocessing


class LazyExperiment:
    def __init__(self, experiment=None, experiment_base_path=None):
        self.experiment = experiment
        self.experiment_base_path = experiment_base_path

    def get(self):
        if self.experiment:
            return self.experiment
        elif self.experiment_base_path:
            experiment_pickle_path = os.path.join(self.experiment_base_path, 'config.pickle')
            return experiment_rewrite.BathroomTileExperiment.load_experiment_config(experiment_pickle_path)
        else:
            raise ValueError("One of the two must be specified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed experiment generation')
    parser.add_argument('--shards', dest='shards', type=int, help='Shards of the experiment to run using this machine',
                        nargs='+', default=[0])
    parser.add_argument('--nshards', dest='nshards', type=int, help='Total number of shards', default=1)
    parser.add_argument('--base_dir', type=str, default='/tmp/experiment')
    parser.add_argument('--run', action='store_true', default=False)
    parser.add_argument('--analyze', action='store_true', default=False)
    parser.add_argument('--reanalyze', action='store_true', default=False)

    parsed_args = parser.parse_args()

    base_directory = parsed_args.base_dir
    pickle_path = os.path.join(base_directory, "configs.pickle")

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            experiment_paths = pickle.load(f)
        experiment_gen = enumerate(LazyExperiment(experiment_base_path=basedir) for basedir in experiment_paths)
    else:
        experiment_gen = enumerate(LazyExperiment(experiment) for experiment in experiment_generator(base_directory))

    if parsed_args.run:
        if not os.path.exists(base_directory):
            print("Making directory: {}".format(base_directory))
            os.makedirs(base_directory)

        nshards = parsed_args.nshards
        shards = parsed_args.shards
        experiments_to_run = ((i, exp.get()) for i, exp in experiment_gen if i % nshards in shards)

        def f(i_exp):
            i, experiment = i_exp
            print("Running experiment: {}".format(i))
            experiment_dir = os.path.join(base_directory, "experiment_{}".format(i))
            experiment_data_path = os.path.join(experiment_dir, "data")

            # Runs experiment and saves results.
            experiment.run_experiment_or_load(experiment_data_path)
            experiment.save_experiment_config(os.path.join(experiment_dir, 'config.pickle'))
            print("Done with experiment: {}".format(i))
            return experiment_dir

        # Use a map to stop dwave memory leaks from killing the process.
        mp = multiprocessing.Pool(1)
        experiment_paths = mp.map(f, experiments_to_run)
        mp.close()
        experiment_gen = enumerate(LazyExperiment(experiment_base_path=basedir) for basedir in experiment_paths)

    if parsed_args.analyze or parsed_args.reanalyze:
        experiment_paths = []

        scalars = collections.defaultdict(dict)
        for i, experiment in experiment_gen:
            print("Analyzing experiment {}".format(i))
            experiment_dir = os.path.join(base_directory, "experiment_{}".format(i))
            experiment_paths.append(experiment_dir)
            scalars_path = os.path.join(experiment_dir, 'scalars.pickle')

            if not parsed_args.reanalyze and os.path.exists(scalars_path):
                with open(scalars_path, 'rb') as f:
                    exp_scalars = pickle.load(f)
            else:
                experiment = experiment.get()

                data, energies, exp_scalars = experiment.run_experiment_or_load(os.path.join(experiment_dir, 'data'))
                analyzer = bathroom_tile.graphanalysis.GraphAnalyzer(experiment.graph, experiment.graph.all_vars,
                                                                     data, energies)

                orientations = analyzer.calculate_difference_order_parameter()
                orientation_count = numpy.mean(orientations)
                abs_orientation_count = numpy.mean(numpy.abs(orientations))
                exp_scalars['orientation_count'] = orientation_count
                exp_scalars['abs_orientation_count'] = abs_orientation_count

                flippables = analyzer.get_flippable_squares()
                flippable_count = numpy.mean(numpy.sum(flippables, axis=0))
                exp_scalars['flippable_count'] = flippable_count

                with open(scalars_path, 'wb') as w:
                    pickle.dump(exp_scalars, w)

            for k, v in exp_scalars.items():
                scalars[k][i] = v

        if not os.path.exists(pickle_path):
            print("Attempting to save experiment configurations to {}".format(pickle_path))
            with open(pickle_path, "wb") as w:
                pickle.dump(experiment_paths, w)
                print("\tDone!")

        scalar_keys = set()
        for k, v in scalars.items():
            for i in v:
                scalar_keys.add(i)

        scalar_clean = collections.defaultdict(list)
        for i in sorted(scalar_keys):
            if all(i in scalars[k] for k in scalars):
                for k in scalars:
                    scalar_clean[k].append(scalars[k][i])

        for k, vs in scalars.items():
            with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
                for i, v in sorted(vs.items()):
                    w.write("{}\t{}\n".format(i, v))
            i_s, v_s = zip(*sorted(vs.items()))
            pyplot.plot(i_s, v_s, 'x--')
            pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
            pyplot.clf()

        plot_functions = [flippable_phase, orientation_phase]
        for plot_fn in plot_functions:
            try:
                plot_fn(base_directory, scalar_clean)
            except Exception as e:
                print("Could not plot using {}".format(plot_fn))
                print(e)
