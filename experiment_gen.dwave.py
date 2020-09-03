from experiment_rewrite import BathroomTileExperiment
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    ng = 20
    nj = 20
    beta = 39.72

    num_reads = 10000

    mtfh = dwave_sampler.MachineTransverseFieldHelper()

    periodic_j = 1.0
    periodic_factor = 10.0
    js = 1./numpy.linspace(beta, 1.0, nj)
    if periodic_j:
        js = js / periodic_factor

    gamma_ratios = 10**numpy.linspace(-2, 1, ng-1)
    gamma_ratios = numpy.asarray([0] + list(gamma_ratios))
    JS, GS = numpy.meshgrid(js, gamma_ratios)
    samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    for i, (j, gamma_ratio) in enumerate(samples_to_take):
        if gamma_ratio > 0:
            s = mtfh.s_for_gamma_ratio(gamma_ratio*j)
            abs_gamma = mtfh.gamma_for_s(s)
            j_mult = mtfh.j_for_s(s)
        else:
            abs_gamma = gamma_ratio
            j_mult = 1.0

        experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
        config_pickle = os.path.join(experiment_dir, 'config.pickle')
        if os.path.exists(config_pickle):
            experiment = BathroomTileExperiment.load_experiment_config(config_pickle)
        else:
            min_x = 7 + 1
            max_x = 15 - 1
            min_y = 0 + 1
            max_y = 8 - 1
            unit_cell_rect = ((min_x, max_x), (min_y, max_y))
            experiment = BathroomTileExperiment(dwave_sampler.DWaveSampler,
                                                unit_cell_rect=unit_cell_rect,
                                                dwave_periodic_j=periodic_j,
                                                base_ej_over_kt=beta * j_mult,
                                                j=j, gamma=abs_gamma, num_reads=num_reads,
                                                # sampler_build_kwargs={'dry_run': 1.0}
                                                )
        yield experiment
