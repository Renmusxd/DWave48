import experiment_rewrite
import pickle
import os
import sys


if __name__ == "__main__":
    experiment_group_base = sys.argv[1]
    new_experiment_directory = sys.argv[2]

    print(experiment_group_base, new_experiment_directory)

    with open(os.path.join(experiment_group_base, 'configs.pickle'), 'rb') as f:
        configs = pickle.load(f)

    def foo(experiment_path):
        dir_to_make = experiment_path[len(experiment_group_base)+1:]
        return os.path.join(new_experiment_directory, dir_to_make)

    experiment_paths = [config.base_dir for config in configs]
    new_experiment_paths = [foo(config.base_dir) for config in configs]

    if not os.path.exists(new_experiment_directory):
        os.makedirs(new_experiment_directory)

    with open(os.path.join(new_experiment_directory, 'configs.pickle'), 'wb') as w:
        pickle.dump(new_experiment_paths, w)

    print("Processing:")
    for experiment_path in experiment_paths:
        print("\t{}".format(experiment_path))
        with open(os.path.join(experiment_path, 'config.pickle'), 'rb') as f:
            config = pickle.load(f)
        experiment = experiment_rewrite.BathroomTileExperiment(config.sampler_fn, graph=config.graph,
                                                               base_ej_over_kt=config.ej_over_kt,
                                                               j=config.j, gamma=config.gamma,
                                                               num_reads=config.num_reads,
                                                               sampler_build_kwargs=config.sampler_kwargs,
                                                               sampler_sample_kwargs=config.sample_kwargs)

        dir_to_make = experiment_path[len(experiment_group_base)+1:]
        new_dir = os.path.join(new_experiment_directory, dir_to_make)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            os.makedirs(os.path.join(new_dir, 'data'))

        experiment.save_experiment_config(os.path.join(new_dir, 'config.pickle'))
        if config.response is not None:

            scalars = {
                "ej_by_kt": experiment.j*experiment.base_ej_over_kt,
                "j": experiment.j,
                "gamma": experiment.gamma
            }
            scalars.update(config.response.scalars())

            experiment.save_data(os.path.join(new_dir, 'data'),
                                 config.response.data(),
                                 config.response.energy(),
                                 scalars)
