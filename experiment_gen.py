from bathroom_tile.experiments import *
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    n = 10
    # betas = 10**numpy.linspace(-2, 0, n)
    beta = 39.72
    h = 0.0
    j = 1.0
    gammas = numpy.linspace(0.01, 0.5, n)
    mapper = dwave_sampler.MachineTransverseFieldHelper()
    for i, gamma in enumerate(gammas):
        tmp_s = mapper.s_for_gamma(gamma)
        tmp_j = j*mapper.j_for_s(tmp_s)
        print("Building experiment {}".format(i))
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir,
                                  quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                  h=h, j=tmp_j, gamma=gamma, throw_errors=True,
                                  ej_over_kt=beta,
                                  sampler_kwargs={'beta': beta,
                                                  'thermalization_time': 1e4,
                                                  'timesteps': 1e7 + 1e4,
                                                  'sampling_freq': 1e4},
                                  )
        config.num_reads = 10000
        config.auto_scale = False
        config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8) #, build_kwargs={'ideal_periodic_boundaries': True})
        yield config
