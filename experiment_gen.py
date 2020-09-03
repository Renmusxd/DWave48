from experiment_rewrite import BathroomTileExperiment
from bathroom_tile import quantum_monte_carlo_simulator
import os
import numpy


def experiment_generator(base_dir):
    ng = 10
    nj = 10
    beta = 100.0

    num_experiments = 8
    num_reads = 10000
    therm_time = int(1e5)
    run_time = int(1e5)

    sampling_freq = num_experiments*(run_time // num_reads)

    # mtfh = dwave_sampler.MachineTransverseFieldHelper()

    js = 1./numpy.linspace(10*beta, 1.0, nj)
    gamma_ratios = 10**numpy.linspace(-2, numpy.log10(3), ng)
    JS, GS = numpy.meshgrid(js, gamma_ratios)
    samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    for i, (j, gamma_ratio) in enumerate(samples_to_take):
        # s = mtfh.s_for_gamma_ratio(gamma_ratio*j)
        # abs_gamma = mtfh.gamma_for_s(s)
        # j_mult = mtfh.j_for_s(s)

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
                                                j=j, gamma=gamma_ratio*j, num_reads=num_reads,
                                                graph_build_kwargs={'ideal_periodic_boundaries': True},
                                                sampler_build_kwargs={'thermalization_time': therm_time,
                                                                      'timesteps': therm_time + run_time,
                                                                      'sampling_freq': sampling_freq,
                                                                      'rvb': True},
                                                # sampler_build_kwargs={'dry_run': 1.0}
                                                )
        yield experiment