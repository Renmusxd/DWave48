from experiment_rewrite import BathroomTileExperiment
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    ng = 20
    # nj = 5
    beta = 39.72

    num_reads = 10
    therm_time = int(1e8)
    run_time = int(1e8)
    sampling_freq = run_time // num_reads

    # mtfh = dwave_sampler.MachineTransverseFieldHelper()

    # js = 1./numpy.linspace(beta, 1.0, nj)
    j = 1.0
    gamma_ratios = numpy.linspace(0.05, 5.0, ng)
    # JS, GS = numpy.meshgrid(js, gamma_ratios)
    # samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    # for i, (j, gamma_ratio) in enumerate(samples_to_take):
    for i, gamma_ratio in enumerate(gamma_ratios):
        # s = mtfh.s_for_gamma_ratio(gamma_ratio*j)
        # abs_gamma = mtfh.gamma_for_s(s)
        # j_mult = mtfh.j_for_s(s)

        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        config_pickle = os.path.join(experiment_dir, 'config.pickle')
        if os.path.exists(config_pickle):
            experiment = BathroomTileExperiment.load_experiment_config(config_pickle)
        else:
            min_x = 0
            max_x = 15
            min_y = 0
            max_y = 15
            unit_cell_rect = ((min_x, max_x), (min_y, max_y))
            experiment = BathroomTileExperiment(quantum_monte_carlo_simulator.QuantumMonteCarloSampler,
                                                unit_cell_rect=unit_cell_rect,
                                                base_ej_over_kt=beta,
                                                j=j, gamma=gamma_ratio, num_reads=num_reads,
                                                graph_build_kwargs={'ideal_periodic_boundaries': True},
                                                sampler_build_kwargs={'thermalization_time': therm_time,
                                                                      'timesteps': therm_time + run_time,
                                                                      'sampling_freq': sampling_freq},
                                                # sampler_build_kwargs={'dry_run': 1.0}
                                                )
        yield experiment
