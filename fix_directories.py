import pickle
import os


if __name__ == "__main__":
    experiment_name = "data/full_phase_diagram_sweep_qmc"

    configs_file = os.path.join(experiment_name, "configs.pickle")
    if os.path.exists(configs_file):
        with open(configs_file, 'rb') as f:
            configs = pickle.load(f)

        for i,config in enumerate(configs):
            print("Old:\t{}".format(config.base_dir))
            config.base_dir = os.path.join(experiment_name, "experiment_{}".format(i))
            print("New:\t{}".format(config.base_dir))

            with open(os.path.join(config.base_dir, 'config.pickle'), 'rb') as f:
                config_full = pickle.load(f)

            config_full.base_dir = os.path.join(experiment_name, "experiment_{}".format(i))
            config_full.save_self(os.path.join(config.base_dir, "config.pickle"))

    with open(configs_file, 'wb') as w:
        pickle.dump(configs, w)
