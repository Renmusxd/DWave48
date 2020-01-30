import tempfile
import graphbuilder
import graphanalysis
import monte_carlo_simulator
import main
from matplotlib import pyplot


def monte_carlo_sampler_fn():
    return monte_carlo_simulator.MonteCarloSampler(read_all_energies=True)


def make_energies_fn(config):
    all_energies = config.response.all_energies
    for es in all_energies:
        pyplot.plot(es)
    pyplot.xscale('log')
    pyplot.show()

if __name__ == "__main__":
    tdir = tempfile.mkdtemp()
    config = main.ExperimentConfig(tdir, monte_carlo_sampler_fn, h=0.0, j=1.0)
    config.num_reads = 10
    config.auto_scale = False
    config.build_graph(min_x=0, max_x=8, min_y=0, max_y=8)
    config.run_or_load_experiment()
    config.add_meta_analysis(make_energies_fn)
    results = config.analyze()