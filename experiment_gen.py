from experiments import *
from dwave.system.samplers import DWaveSampler
import monte_carlo_simulator
import quantum_monte_carlo_simulator
import mocksampler
import os


def experiment_generator(base_dir):
    n = 3
    for i in range(1, n+1):
        print("Building experiment {}".format(i))
        h = 0.0  # float(i) / n
        j = 1.0
        transverse = float(i) / n
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir,
                                  quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                  sampler_kwargs={'beta': 1.0,
                                                  'thermalization_time': 1e3,
                                                  'timesteps': 1e5,
                                                  'sampling_freq': 1e3},
                                  h=h, j=j, gamma=transverse, throw_errors=True)
        config.num_reads = 8
        config.auto_scale = False
        # config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8)
        config.build_graph(min_x=0, max_x=3, min_y=0, max_y=3)
        yield config
