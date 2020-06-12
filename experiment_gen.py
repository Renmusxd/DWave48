from bathroom_tile.experiments import *
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    ng = 20
    nj = 20
    beta = 39.72
    h = 0.0
    gammas = 10**numpy.linspace(-2, -1, ng)
    js = 10**numpy.linspace(numpy.log10(1/beta), 0, nj)
    JS, GS = numpy.meshgrid(js, gammas)
    samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    mapper = dwave_sampler.MachineTransverseFieldHelper()
    for i, (j, gamma) in enumerate(samples_to_take):
        tmp_s = mapper.s_for_gamma(gamma)
        tmp_j = j #/ mapper.j_for_s(tmp_s)
        print("Building experiment {}".format(i))
        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        print("\tUsing directory: {}".format(experiment_dir))
        config = ExperimentConfig(experiment_dir,
                                  dwave_sampler.DWaveSampler,
                                  h=h, j=tmp_j, gamma=gamma, throw_errors=True,
                                  ej_over_kt=tmp_j*beta,
                                  # sampler_kwargs={'dry_run': True}
                                  # sampler_kwargs={'beta': beta,
                                  #                 'thermalization_time': 1e6,
                                  #                 'timesteps': 1e6 + 1e6,
                                  #                 'sampling_freq': 1e3},
                                  )
        config.num_reads = 10000
        config.auto_scale = False
        config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8) #, build_kwargs={'ideal_periodic_boundaries': True})
        yield config
