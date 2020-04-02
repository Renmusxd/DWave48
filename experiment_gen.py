from experiments import *
from dwave.system.samplers import DWaveSampler
import monte_carlo_simulator
import quantum_monte_carlo_simulator
import mocksampler
import os
import numpy


def experiment_generator(base_dir):
    n = 20
    betas = 10**numpy.linspace(-2,0, n)
    h = 0.0
    j = 1.0
    gamma = 1.0
    for i, beta in enumerate(betas):
        print("Building experiment {}".format(i))
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir,
                                  quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                  sampler_kwargs={'beta': beta,
                                                  'thermalization_time': 1e6,
                                                  'timesteps': 1e7 + 1e6,
                                                  'sampling_freq': 1e3},
                                  h=h, j=j, gamma=gamma, throw_errors=True)
        config.num_reads = 10000
        config.auto_scale = False
        config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8, build_kwargs={'ideal_periodic_boundaries': True})
        yield config
