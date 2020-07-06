from experiment_rewrite import BathroomTileExperiment
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    num_reads = 10000

    ng = 20
    nj = 10
    beta = 39.72

    mtfh = dwave_sampler.MachineTransverseFieldHelper()

    gamma_ratios = numpy.linspace(0.01/beta, 3.0/beta, ng)
    js = 1./numpy.linspace(beta, 1.0, nj)
    JS, GS = numpy.meshgrid(js, gamma_ratios)
    samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    for i, (j, gamma_ratio) in enumerate(samples_to_take):
        s = mtfh.s_for_gamma_ratio(gamma_ratio)
        abs_gamma = mtfh.gamma_for_s(s)
        j_mult = mtfh.j_for_s(s)

        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        config_pickle = os.path.join(experiment_dir, 'config.pickle')
        if os.path.exists(config_pickle):
            experiment = BathroomTileExperiment.load_experiment_config(config_pickle)
        else:
            min_x = 7
            max_x = 15
            min_y = 0
            max_y = 8
            unit_cell_rect = ((min_x, max_x), (min_y, max_y))
            experiment = BathroomTileExperiment(quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                                unit_cell_rect=unit_cell_rect,
                                                base_ej_over_kt=beta,
                                                j=j, gamma=abs_gamma, num_reads=num_reads,
                                                graph_build_kwargs={'ideal_periodic_boundaries': True},
                                                sampler_build_kwargs={'thermalization_time': 1e6,
                                                                      'timesteps': 1e7 + 1e6,
                                                                      'sampling_freq': 1e4},
                                                # sampler_build_kwargs={'dry_run': True}
                                                )
        yield experiment
