
from experiment_gen import experiment_generator
from experiment_metaanalysis import *
import argparse
import os
import pickle
import collections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed experiment generation')
    parser.add_argument('--shards', dest='shards', type=int, help='Shards of the experiment to run using this machine',
                        nargs='+')
    parser.add_argument('--base_dir', type=str, default='/tmp/experiment')
    parser.add_argument('--analyze', action='store_true', default=False)
    parsed_args = parser.parse_args()

    base_directory = parsed_args.base_dir
    pickle_path = os.path.join(base_directory, "configs.pickle")
    experiment_gen = experiment_generator(base_directory)

    configs = [config for config in experiment_gen]
    if not os.path.exists(base_directory):
        print("Making directory: {}".format(base_directory))
        os.makedirs(base_directory)

    if parsed_args.analyze:
        if not os.path.exists(pickle_path):
            print("Attempting to save experiment configurations to {}".format(pickle_path))
            with open(pickle_path, "wb") as w:
                pickle.dump(configs, w)
                print("\tDone!")
    
    shards = parsed_args.shards or list(range(len(configs)))
    experiments_to_run = (exp for i, exp in enumerate(configs)
                          if i in shards)

    scalars = collections.defaultdict(list)
    for experiment in experiments_to_run:
        experiment.run_or_load_experiment()
        if parsed_args.analyze:
            experiment.add_default_analyzers()
            results = experiment.analyze()
            for k, v in results.items():
                scalars[k].append(v)

    if parsed_args.analyze:
        for k, vs in scalars.items():
            with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
                for i, v in enumerate(vs):
                    w.write("{}\t{}\n".format(i, v))

            pyplot.plot(vs, 'x--')
            pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
            pyplot.clf()

        plot_functions = [defect_plot, flippable_plot, unit_cell_divergence_plot]
        for plot_fn in plot_functions:
            try:
                plot_fn(base_directory, scalars)
            except Exception as e:
                print("Could not plot using {}".format(plot_fn))
                print(e)
