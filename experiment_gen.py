from experiment_rewrite import BathroomTileExperiment
from bathroom_tile import quantum_monte_carlo_simulator
from bathroom_tile import dwave_sampler
import os
import numpy


def experiment_generator(base_dir):
    num_reads = 1000

    ng = 20
    nj = 20
    beta = 39.72
    gammas = 10**numpy.linspace(-2, -1, ng)
    js = 10**numpy.linspace(numpy.log10(1/beta), 0, nj)
    JS, GS = numpy.meshgrid(js, gammas)
    samples_to_take = list(zip(JS.flatten(), GS.flatten()))

    for i, (j, gamma) in enumerate(samples_to_take):
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
            experiment = BathroomTileExperiment(dwave_sampler.DWaveSampler, unit_cell_rect=unit_cell_rect,
                                                base_ej_over_kt=beta,
                                                j=j, gamma=gamma*j, num_reads=num_reads)
        yield experiment
