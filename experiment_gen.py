from experiments import *
from dwave.system.samplers import DWaveSampler
import monte_carlo_simulator
import quantum_monte_carlo_simulator
import mocksampler
import os


def experiment_generator(base_dir):
    n = 10
    for i in range(1, n+1):
        print("Building experiment {}".format(i))
        h = 0.0  # float(i) / n
        j = 1.0
        transverse = float(i) / n
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir,
                                  quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                  sampler_kwargs={'beta': 39.72,
                                                  'thermalization_time': 1e7,
                                                  'timesteps': 1e9 + 1e7,
                                                  'sampling_freq': 1e6},
                                  h=h, j=j, gamma=transverse, throw_errors=True)
        config.num_reads = 10000
        config.auto_scale = False
        config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8)
        yield config
