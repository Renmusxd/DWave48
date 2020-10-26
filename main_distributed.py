from bathroom_tile.experiment_metaanalysis import flippable_phase, orientation_phase, psi_phase, gl_phase, s_phase
from bathroom_tile import graphbuilder
from experiment_gen import experiment_generator
import experiment_rewrite
import bathroom_tile
from matplotlib import pyplot
import numpy
import argparse
import os
import pickle
import collections
import multiprocessing


class LazyExperiment:
    def __init__(self, experiment=None, experiment_base_path=None):
        self.experiment = experiment
        self.experiment_base_path = experiment_base_path

    def get(self):
        if self.experiment:
            return self.experiment
        elif self.experiment_base_path:
            print(self.experiment_base_path)
            experiment_pickle_path = os.path.join(self.experiment_base_path, 'config.pickle')
            return experiment_rewrite.BathroomTileExperiment.load_experiment_config(experiment_pickle_path)
        else:
            raise ValueError("One of the two must be specified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed experiment generation')
    parser.add_argument('--shards', dest='shards', type=int, help='Shards of the experiment to run using this machine',
                        nargs='+', default=[0])
    parser.add_argument('--nshards', dest='nshards', type=int, help='Total number of shards', default=1)
    parser.add_argument('--base_dir', type=str, default='/tmp/experiment')
    parser.add_argument('--run', action='store_true', default=False)
    parser.add_argument('--analyze', action='store_true', default=False)
    parser.add_argument('--reanalyze', action='store_true', default=False)
    parser.add_argument('--multiprocess_hack', action='store_true', default=False)

    parsed_args = parser.parse_args()

    base_directory = parsed_args.base_dir
    pickle_path = os.path.join(base_directory, "configs.pickle")

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            experiment_paths = pickle.load(f)
        experiment_gen = enumerate(LazyExperiment(experiment_base_path=basedir) for basedir in experiment_paths)
    else:
        experiment_gen = enumerate(LazyExperiment(experiment) for experiment in experiment_generator(base_directory))

    if parsed_args.run:
        if not os.path.exists(base_directory):
            print("Making directory: {}".format(base_directory))
            os.makedirs(base_directory)

        nshards = parsed_args.nshards
        shards = parsed_args.shards
        experiments_to_run = ((i, exp.get()) for i, exp in experiment_gen if i % nshards in shards)

        def f(i_exp):
            i, experiment = i_exp
            print("Running experiment: {}".format(i))
            experiment_dir = os.path.join(base_directory, "experiment_{}".format(i))
            experiment_data_path = os.path.join(experiment_dir, "data")

            # Runs experiment and saves results.
            data, energies, scalars = experiment.run_experiment_or_load(experiment_data_path)
            experiment.save_experiment_config(os.path.join(experiment_dir, 'config.pickle'))
            experiment.draw_min_energy_dimers(data, energies, experiment_dir)
            print("Done with experiment: {}".format(i))
            return experiment_dir

        # Use a map to stop dwave memory leaks from killing the process.
        if parsed_args.multiprocess_hack:
            mp = multiprocessing.Pool(1)
            experiment_paths = mp.map(f, experiments_to_run)
            mp.close()
        else:
            experiment_paths = list(map(f, experiments_to_run))
        experiment_gen = enumerate(LazyExperiment(experiment_base_path=basedir) for basedir in experiment_paths)

    if parsed_args.analyze or parsed_args.reanalyze:
        experiment_paths = []

        scalars = collections.defaultdict(dict)
        for i, experiment in experiment_gen:
            print("Analyzing experiment {}".format(i))
            experiment_dir = os.path.join(base_directory, "experiment_{}".format(i))
            experiment_paths.append(experiment_dir)
            scalars_path = os.path.join(experiment_dir, 'scalars.pickle')

            if not parsed_args.reanalyze and os.path.exists(scalars_path):
                with open(scalars_path, 'rb') as f:
                    exp_scalars = pickle.load(f)
            else:
                experiment = experiment.get()
                if experiment:
                    bathroom_tile.draw_dwave.draw_graph(experiment.graph, os.path.join(experiment_dir, 'chimera.svg'))
                    result = experiment.run_experiment_or_load(os.path.join(experiment_dir, 'data'), allow_run=False)
                    if result is not None:
                        data, energies, exp_scalars = result
                        analyzer = bathroom_tile.graphanalysis.GraphAnalyzer(experiment.graph, experiment.graph.all_vars,
                                                                             data, energies)

                        # orientations = analyzer.calculate_difference_order_parameter()
                        # orientation_count = numpy.mean(orientations)
                        # abs_orientation_count = numpy.mean(numpy.abs(orientations))
                        # exp_scalars['orientation_count'] = orientation_count
                        # exp_scalars['abs_orientation_count'] = abs_orientation_count
                        #
                        # flippables = analyzer.get_flippable_squares()
                        # flippable_count = numpy.mean(numpy.sum(flippables, axis=0))
                        # exp_scalars['flippable_count'] = flippable_count
                        #
                        # complex_pi_pi = analyzer.calculate_complex_angle_order_parameter()
                        # complex_pi_pi_average = numpy.mean((complex_pi_pi**4).real)
                        # exp_scalars['complex_pi_pi_average'] = complex_pi_pi_average
                        #
                        # # Structure factor
                        # # Get diagonal unit cell distance
                        # x0, y0 = graphbuilder.get_unit_cell_cartesian(0,0)
                        # x1, y1 = graphbuilder.get_unit_cell_cartesian(1,1)
                        # r = numpy.sqrt((x1-x0)**2 + (y1-y0)**2)
                        #
                        # ks = numpy.linspace(-numpy.pi, numpy.pi, 17)  # -8pi/8 .. + 0
                        # kas = numpy.dstack([ks, ks])
                        #
                        # KAs, KBs = numpy.meshgrid(ks, ks)
                        # KAs_xy = numpy.dstack([KAs, KAs])/r
                        # KBs_xy = numpy.dstack([KBs, -KBs])/r
                        #
                        # orders = analyzer.calculate_fourier_order_parameter(k_nesw=KAs_xy, k_nwse=KBs_xy)
                        # orders = numpy.mean(orders, axis=-1)
                        # fourier_order = numpy.sum(orders, axis=0)
                        #
                        # try:
                        #     pyplot.contourf(KAs, KBs, fourier_order)
                        #     pyplot.colorbar()
                        #     pyplot.xlabel(r'k(1,1)')
                        #     pyplot.ylabel(r'k(1,-1)')
                        #     pyplot.savefig(os.path.join(experiment_dir, 'fourier_order.png'))
                        #     pyplot.close()
                        # except Exception as e:
                        #     print(e)
                        #
                        # pi_pi_order_params = orders[:,-1,-1]
                        # psi = 2*(pi_pi_order_params[0] + 1.0j*pi_pi_order_params[1])
                        # order_param = numpy.abs(psi) * numpy.cos(4 * numpy.angle(psi))
                        # exp_scalars['psi_order_param'] = order_param

                        order, abs_order = analyzer.calculate_gl_order_parameter()
                        pyplot.scatter(order.real, order.imag, alpha=0.1)
                        t = numpy.linspace(0, 2 * numpy.pi, 360)

                        pyplot.plot(numpy.cos(t), numpy.sin(t), c='b')
                        pyplot.savefig(os.path.join(experiment_dir, "gl_order.png"))
                        pyplot.close()

                        angles = numpy.angle(order)
                        bins = numpy.linspace(-numpy.pi, numpy.pi, 360)
                        s_bins = numpy.expand_dims(bins, axis=-1)
                        min_dist_2 = numpy.abs([angles - s_bins,
                                                (angles + 2*numpy.pi) - s_bins,
                                                (angles - 2*numpy.pi) - s_bins])**2
                        min_dist_2 = numpy.min(min_dist_2, axis=0)
                        sig = 2*numpy.pi / 360
                        d_exp = numpy.exp( - min_dist_2 / (2.0*sig))
                        d_exp = d_exp / numpy.sum(d_exp, axis=0)
                        blurred = numpy.mean(d_exp, axis=-1) * len(bins)

                        pyplot.plot(bins / numpy.pi, blurred, 'x-')
                        pyplot.grid()
                        ylow, yhigh = pyplot.ylim()
                        pyplot.ylim((0, yhigh))
                        pyplot.xlabel(r'$\theta / \pi$')
                        pyplot.ylabel('A label to make Matteo happy')
                        pyplot.savefig(os.path.join(experiment_dir, "gl_hist.png"))
                        pyplot.close()

                        mu = numpy.mean(blurred)
                        pyplot.plot(mu*numpy.cos(bins), mu*numpy.sin(bins), c='b')
                        pyplot.plot(blurred*numpy.cos(bins), blurred*numpy.sin(bins), c='r')
                        pyplot.grid()
                        pyplot.savefig(os.path.join(experiment_dir, 'gl_polar_hist.png'))
                        pyplot.close()

                        exp_scalars['gl_order'] = numpy.mean(order)
                        exp_scalars['abs_gl_order'] = numpy.mean(abs_order)

                        with open(scalars_path, 'wb') as w:
                            pickle.dump(exp_scalars, w)
                    else:
                        print("No response {}".format(i))
                        exp_scalars = {}
                else:
                    print("No experiment {}".format(i))
                    exp_scalars = {}

            for k, v in exp_scalars.items():
                scalars[k][i] = v

        if not os.path.exists(pickle_path):
            print("Attempting to save experiment configurations to {}".format(pickle_path))
            with open(pickle_path, "wb") as w:
                pickle.dump(experiment_paths, w)
                print("\tDone!")

        scalar_keys = set()
        for k, v in scalars.items():
            for i in v:
                scalar_keys.add(i)

        scalar_clean = collections.defaultdict(list)
        for i in sorted(scalar_keys):
            if all(i in scalars[k] for k in scalars):
                for k in scalars:
                    scalar_clean[k].append(scalars[k][i])

        for k, vs in scalars.items():
            with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
                for i, v in sorted(vs.items()):
                    w.write("{}\t{}\n".format(i, v))
            i_s, v_s = zip(*sorted(vs.items()))
            pyplot.plot(i_s, v_s, 'x--')
            pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
            pyplot.clf()

        with open(os.path.join(base_directory, "scalars.pickle"), 'wb') as w:
            pickle.dump(scalars, w)

        plot_functions = [gl_phase]
        for plot_fn in plot_functions:
            try:
                plot_fn(base_directory, scalar_clean)
            except Exception as e:
                print("Could not plot using {}".format(plot_fn))
                print(e)
