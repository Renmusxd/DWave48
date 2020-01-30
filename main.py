from dwave.system.samplers import DWaveSampler
import graphbuilder
import graphanalysis
import graphdrawing
import monte_carlo_simulator
import pickle
import os
import collections
from matplotlib import pyplot
import numpy

from mocksampler import MockSampler


class ExperimentConfig:
    def __init__(self, base_dir, sampler_fn, machine_temp=14.5e-3, h=0.0, j=1.0, build_kwargs=None, sample_kwargs=None):
        self.base_dir = base_dir
        self.sampler_fn = sampler_fn
        self.graph = None
        self.hs = {}
        self.num_reads = 0
        self.auto_scale = True
        self.response = None
        self.data = None
        self.machine_temp = machine_temp
        self.h = h
        self.j = j
        self.build_kwargs = build_kwargs or {}
        self.sample_kwargs = sample_kwargs or {}
        self.analyzers = []
        self.meta_analysis = []

    def build_graph(self, max_x=8, max_y=16, min_x=0, min_y=0, hs_override=None, build_kwargs=None):
        gb = graphbuilder.GraphBuilder(j=self.j)
        gb.add_cells([
            (x, y)
            for x in range(min_x, max_x)
            for y in range(min_y, max_y)
        ])
        gb.connect_all()
        kwargs_for_build = self.build_kwargs or {}
        kwargs_for_build.update(build_kwargs or {})
        self.graph = gb.build(h=self.h, **kwargs_for_build)
        self.hs = self.graph.hs
        if hs_override is not None:
            self.hs.update(hs_override)

    def maybe_load_self(self, filepath):
        if os.path.exists(filepath):
            print("\tLoading self... ", end='')
            try:
                with open(filepath, "rb") as f:
                    config = pickle.load(f)
                backup_graph = self.graph
                self.override(config)
                self.graph = backup_graph
                print("done!")
                return True
            except Exception as e:
                print("error: {}".format(str(e)))
        return False

    def save_self(self, filepath):
        print("\tSaving self... ", end='')
        with open(filepath, "wb") as w:
            pickle.dump(self, w)
        print("done!")

    def override(self, config):
        self.hs = config.hs
        self.num_reads = config.num_reads
        self.auto_scale = config.auto_scale
        self.base_dir = config.base_dir
        self.response = config.response
        self.data = config.data
        self.graph = config.graph

    def run_or_load_experiment(self, sample_kwargs=None):
        filepath = os.path.join(self.base_dir, "config.pickle")
        if not self.maybe_load_self(filepath):
            if self.graph is None:
                raise Exception("Graph not yet built")
            print("Running on dwave... ", end='')
            kwargs_for_sample = self.sample_kwargs or {}
            kwargs_for_sample.update(sample_kwargs or {})
            self.response = self.sampler_fn().sample_ising(self.hs, self.graph.edges,
                                                           num_reads=self.num_reads,
                                                           auto_scale=self.auto_scale,
                                                           **kwargs_for_sample)
            self.data = [({k: sample[k] for k in sample}, energy, num_occurences)
                         for sample, energy, num_occurences in self.response.data()]
            print("done!")
            self.save_self(filepath)

    def add_analysis(self, analyzer_fn):
        self.analyzers.append(analyzer_fn)

    def analyze(self):
        print("Running analysis on {}".format(self.base_dir))

        def color_by_sign(var_a, var_b):
            if self.graph.edges[(var_a, var_b)] < 0:
                return "rgb(0,0,0)"
            else:
                return "rgb(255,0,0)"

        print("\tMaking lattice svgs")
        svg = graphdrawing.make_edges_svg(list(self.graph.edges), color_fn=color_by_sign, front=True)
        if svg:
            with open(os.path.join(self.base_dir, "front_lattice.svg"), "w") as w:
                w.write(svg)
        svg = graphdrawing.make_edges_svg(list(self.graph.edges), color_fn=color_by_sign, front=False)
        if svg:
            with open(os.path.join(self.base_dir, "rear_lattice.svg"), "w") as w:
                w.write(svg)

        result_dict = {
            "j": self.j,
            "inv_j": 1.0 / self.j,
            "h": self.h
        }
        if self.data:
            samples = [sample for sample, _, __ in self.data]
            energies = [energy for _, energy, __ in self.data]
            num_occurrences = [num_occurrences for _, __, num_occurrences in self.data]

            graph_analyzer = graphanalysis.GraphAnalyzer(self.graph, samples, energies, num_occurrences)
            for analyzer_fn in self.analyzers:
                try:
                    output = analyzer_fn(self.base_dir, graph_analyzer)
                    if output:
                        result_dict.update(output)
                except Exception as e:
                    print("Failed to run:\t{}".format(analyzer_fn))
                    print(e)
            for analyzer_fn in self.meta_analysis:
                try:
                    analyzer_fn(self)
                except Exception as e:
                    print("Failed to run:\t{}".format(analyzer_fn))
                    raise e

            print("\tDone!")
        return result_dict

    def add_default_analyzers(self):
        def analyzer(analyzer_fn):
            self.add_analysis(analyzer_fn)
            return analyzer_fn

        @analyzer
        def analyzer_plot_energies(base_dir, graph_analyzer):
            print("\tMaking energy histogram")
            pyplot.hist(graph_analyzer.energies, weights=graph_analyzer.num_occurrences)
            pyplot.savefig(os.path.join(base_dir, "energies.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_count_dimers(base_dir, graph_analyzer):
            print("\tCalculating dimers")
            dimers = graph_analyzer.get_dimer_matrix()
            dimers_per_sample = numpy.sum(dimers == 1, axis=0)
            pyplot.hist(dimers_per_sample, weights=graph_analyzer.num_occurrences)
            pyplot.savefig(os.path.join(base_dir, "dimer_counts.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_draw_dimers(base_dir, graph_analyzer):
            # Then plot the dimers for one of the ground states (or lowest E we found anyway).
            print("\tDrawing dimer svgs")
            lowest_e = numpy.argmin(graph_analyzer.energies)
            sample, energy, num_occurrences = self.data[lowest_e]

            draw_dimers(os.path.join(base_dir, "front_min_energy_dimers.svg"), graph_analyzer.graph.edges, sample,
                        front=True, color_on_orientation=False)
            draw_dimers(os.path.join(base_dir, "front_min_energy_dimers_color.svg"), graph_analyzer.graph.edges, sample,
                        front=True, color_on_orientation=True)

            draw_dimers(os.path.join(base_dir, "rear_min_energy_dimers.svg"), graph_analyzer.graph.edges, sample,
                        front=False, color_on_orientation=False)
            draw_dimers(os.path.join(base_dir, "rear_min_energy_dimers_color.svg"), graph_analyzer.graph.edges, sample,
                        front=False, color_on_orientation=False)

        @analyzer
        def analyzer_calculate_variable_correlations(base_dir, graph_analyzer):
            # Now plot the correlation matrix for the variables, and the correlation as a function of distance.
            print("\tCalculating variable correlations")
            var_corr, _, _, _ = graph_analyzer.calculate_correlation_function()
            pyplot.imshow(var_corr, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(base_dir, "variable_correlations.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_variable_distance_correlations(base_dir, graph_analyzer):
            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            var_corr, distance_corr, _, __ = graph_analyzer.calculate_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Distance (in # edges)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "correlation_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_variable_euclidean_distance_correlations(base_dir, graph_analyzer):
            print("\tCalculating euclidean distance correlations")
            _, euc_distance_corr, _, _ = graph_analyzer.calculate_euclidean_correlation_function()
            average_euc_corrs = numpy.mean(euc_distance_corr, 0)
            stdv_euc_corrs = numpy.sqrt(numpy.var(euc_distance_corr, 0))
            xs = numpy.arange(average_euc_corrs.shape[0])
            pyplot.errorbar(xs, average_euc_corrs, yerr=stdv_euc_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Distance (with edge length=1.0)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "correlation_euclidean_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_dimer_correlations(base_dir, graph_analyzer):
            # Similarly, get the correlation plot for the dimers
            print("\tCalculating dimer correlations")
            dimer_corrs, _, _, _ = graph_analyzer.calculate_dimer_correlation_function()
            pyplot.imshow(dimer_corrs, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(base_dir, "dimer_correlations.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_dimer_distance_correlations(base_dir, graph_analyzer):
            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            dimer_corrs, distance_corr, _, __ = graph_analyzer.calculate_dimer_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (in # edges in dimer-dual)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "dimer_correlation_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_dimer_euclidean_distance_correlations(base_dir, graph_analyzer):
            # Now the distance part, with error bars.
            print("\tCalculating euclidean distance correlations")
            _, euc_dimer_distance_corr, _, _ = graph_analyzer.calculate_euclidean_dimer_correlation_function()
            average_euc_corrs = numpy.mean(euc_dimer_distance_corr, 0)
            stdv_euc_corrs = numpy.sqrt(numpy.var(euc_dimer_distance_corr, 0))
            xs = numpy.arange(average_euc_corrs.shape[0])
            pyplot.errorbar(xs, average_euc_corrs, yerr=stdv_euc_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (with edge length=1.0)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "dimer_correlation_euclidean_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_diagonal_dimer_correlations(base_dir, graph_analyzer):
            # Similarly, get the correlation plot for the dimers
            print("\tCalculating diagonal dimer correlations")
            dimer_corrs, _, _, _ = graph_analyzer.calculate_diagonal_dimer_correlation_function()
            pyplot.imshow(dimer_corrs, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(base_dir, "diagonal_dimer_correlations.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_diagonal_dimer_distance_correlations(base_dir, graph_analyzer):
            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            dimer_corrs, distance_corr, _, _ = graph_analyzer.calculate_diagonal_dimer_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (in # edges in dimer-dual)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "diagonal_dimer_correlation_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_diagonal_dimer_euclidean_distance_correlations(base_dir, graph_analyzer):
            # Now the distance part, with error bars.
            print("\tCalculating euclidean distance correlations")
            _, euc_dimer_distance_corr, _, _ = graph_analyzer.calculate_euclidean_diagonal_dimer_correlation_function()
            average_euc_corrs = numpy.mean(euc_dimer_distance_corr, 0)
            stdv_euc_corrs = numpy.sqrt(numpy.var(euc_dimer_distance_corr, 0))
            xs = numpy.arange(average_euc_corrs.shape[0])
            pyplot.errorbar(xs, average_euc_corrs, yerr=stdv_euc_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (with edge length=1.0)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "diagonal_dimer_correlation_euclidean_distance.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_oriented_dimer_correlations(base_dir, graph_analyzer):
            print("\tCalculating oriented dimer correlations")
            corrs, _ = graph_analyzer.calculate_oriented_dimer_correlation_function()
            fig, axs = pyplot.subplots(2, 2)
            names = ["nesw-nesw", "nesw-nwse", "nwse-nesw", "nwse-nwse"]
            for name, corr, ax in zip(names, corrs, axs.reshape(-1)):
                average_corr = numpy.mean(corr, 0)
                stdv_corr = numpy.sqrt(numpy.var(corr, 0))
                xs = numpy.arange(average_corr.shape[0])
                ax.errorbar(xs, average_corr, yerr=stdv_corr, label="Average")
                ax.legend()
                ax.grid()
                ax.set_title("{}".format(name))
                ax.set_xlabel("Dimer distance (with edge length=1.0)")
                ax.set_ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "oriented_euclidean_correlations.svg"))
            pyplot.clf()

        @analyzer
        def analyzer_calculate_average_dimer_occupations(base_dir, graph_analyzer):
            # Get the average dimer occupations
            print("\tDrawing dimer occupations")
            average_dimers, stdv_dimers = draw_occupations(os.path.join(base_dir, "dimer_occupation_graph.svg"),
                                                           self.graph.edges, graph_analyzer)
            average_dimers, stdv_dimers = draw_occupations(os.path.join(base_dir, "dimer_occupation_graph_scaled.svg"),
                                                           self.graph.edges, graph_analyzer, scale=True)
            divergence = draw_average_unit_cell_directions(os.path.join(base_dir, "dimer_biases.svg"),
                                                           graph_analyzer)

            # Sort them
            average_dimers, stdv_dimers = zip(*sorted(zip(average_dimers, stdv_dimers), key=lambda x: x[0]))
            average_dimers = numpy.asarray(average_dimers)
            stdv_dimers = numpy.asarray(stdv_dimers)
            xs = numpy.arange(len(average_dimers))
            pyplot.plot(xs, average_dimers)
            pyplot.plot(xs, average_dimers + stdv_dimers, 'r--')
            pyplot.plot(xs, average_dimers - stdv_dimers, 'r--')
            pyplot.savefig(os.path.join(base_dir, "dimer_occupation_plot.svg"))
            pyplot.clf()

            pyplot.hist(average_dimers)
            pyplot.xlabel("Dimer occupation frequencies")
            pyplot.ylabel("Counts")
            pyplot.savefig(os.path.join(base_dir, "dimer_occupation_hist.svg"))
            pyplot.clf()

            return {"divergence": divergence}

        @analyzer
        def analyzer_draw_flippable(base_dir, graph_analyzer):
            # Get flippable plaquettes
            print("\tDrawing flippable states")
            lowest_e = numpy.argmin(graph_analyzer.energies)
            sample, energy, num_occurrences = self.data[lowest_e]
            draw_flippable_states(os.path.join(base_dir, "dimer_flippable_plot.svg"), self.graph.edges, sample)

        @analyzer
        def analyzer_get_flippable(base_dir, graph_analyzer):
            print("\tCalculating flippable count")
            flippable_squares = graph_analyzer.get_flippable_squares()
            flippable_count = numpy.mean(numpy.sum(flippable_squares, 0))
            flippable_stdv = numpy.sqrt(numpy.var(numpy.sum(flippable_squares, 0)))
            return {"flippable_count": flippable_count, "flippable_stdv": flippable_stdv}

        @analyzer
        def analyzer_count_defects(base_dir, graph_analyzer):
            # Count the defects
            print("\tCounting defects")
            defects = graph_analyzer.get_defects()
            total_defects_per_sample = numpy.sum(defects, 0)
            average_defects = numpy.mean(total_defects_per_sample, -1)
            stdv_defects = numpy.sqrt(numpy.var(total_defects_per_sample, -1))

            print("\tDefect histograms")
            pyplot.hist(total_defects_per_sample, bins=numpy.arange(0, numpy.max(total_defects_per_sample)))
            pyplot.savefig(os.path.join(base_dir, "defect_histogram.svg"))
            pyplot.clf()

            return {"average_defects": average_defects, "stdv_defects": stdv_defects}

        @analyzer
        def analyzer_defects_correlations(base_dir, graph_analyzer):
            # Get defect correlations
            print("\tDefect correlations")
            defects_corr, defects_corr_function, _, _ = graph_analyzer.calculate_euclidean_defect_correlation_function()
            defects_corr = numpy.nan_to_num(defects_corr)
            pyplot.imshow(defects_corr, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(base_dir, "defect_correlations.svg"))
            pyplot.clf()

            print("\tCalculating distance correlations")
            average_corrs = numpy.mean(defects_corr_function, 0)
            stdv_corrs = numpy.sqrt(numpy.var(defects_corr_function, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Defect distance (with edge length=1.0)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(base_dir, "defect_correlation_distance.svg"))
            pyplot.clf()

    def add_meta_analysis(self, analysis_fn):
        self.meta_analysis.append(analysis_fn)


def make_run_dir(data_dir, run_format="run_{}"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    i = 0
    while os.path.exists(os.path.join(data_dir, run_format.format(i))):
        i += 1
    pathname = os.path.join(data_dir, run_format.format(i))
    os.mkdir(pathname)
    return pathname


def color_on_orientation_fn(var_a, var_b):
    dx_a, dy_a = graphbuilder.calculate_variable_direction(var_a)
    dx_b, dy_b = graphbuilder.calculate_variable_direction(var_b)
    # Vertical and horizontal bonds are green (and rare)
    if dx_a == -dx_b or dy_a == -dy_b:
        return "green"
    if dx_a == dy_b and dy_a == dx_b:
        return "red"
    if dx_a == -dy_b and dy_a == -dx_b:
        return "blue"
    return "purple"


def draw_dimers(filename, graph, sample, front=True, color_on_orientation=True, var_color_fn=None):
    if not color_on_orientation:
        # Basic color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front, var_color_fn=var_color_fn)
        if svg:
            with open(filename, "w") as w:
                w.write(svg)
    else:
        # Orientation color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front, dimer_color_fn=color_on_orientation_fn, var_color_fn=var_color_fn)
        if svg:
            with open(filename, "w") as w:
                w.write(svg)


def draw_occupations(filename, edges, graph_analyzer, front=True, scale=False):
    dimer_matrix = graph_analyzer.get_dimer_matrix()
    average_dimers = numpy.mean(dimer_matrix == 1, -1)
    stdv_dimers = numpy.var(dimer_matrix == 1, -1)

    if scale:
        min_dimer = numpy.min(average_dimers)
        max_dimer = numpy.max(average_dimers)
        dimer_range = max_dimer - min_dimer
        if dimer_range == 0:
            dimer_range = 1
        average_dimers = (average_dimers - min_dimer)/dimer_range

    # Basic color scheme
    def dimer_color_fn(var_a, var_b):
        edge = (min(var_a, var_b), max(var_a, var_b))
        edge_indx = graph_analyzer.graph.edge_lookup[edge]
        average_values = average_dimers[edge_indx]
        red_color = int(average_values * 256)
        return "rgb({},0,0)".format(red_color)

    svg = graphdrawing.make_dimer_contents(edges, front=front, dimer_color_fn=dimer_color_fn)
    svg = graphdrawing.wrap_with_svg(svg)
    if svg:
        with open(filename, "w") as w:
            w.write(svg)

    return average_dimers, stdv_dimers


def draw_average_unit_cell_directions(filename, graph_analyzer, front=True):
    dimer_matrix = graph_analyzer.get_dimer_matrix()
    average_dimers = numpy.mean(dimer_matrix == 1, -1)
    unit_cell_averages = collections.defaultdict(lambda: (0.0, (0.0, 0.0)))
    for vara, varb in graph_analyzer.graph.sorted_edges:
        if graphbuilder.is_front(vara) != front or graphbuilder.is_front(varb) != front:
            continue
        ax, ay, ar = graphbuilder.get_var_traits(vara)
        bx, by, br = graphbuilder.get_var_traits(varb)
        if ax == bx and ay == by:
            x, y = graphbuilder.get_unit_cell_cartesian(ax, ay)
            dax, day = graphbuilder.get_var_cartesian(vara)
            dbx, dby = graphbuilder.get_var_cartesian(varb)
            dx, dy = (dax + dbx)/2.0 - x, (day + dby)/2.0 - y

            weight = average_dimers[graph_analyzer.graph.edge_lookup[(vara, varb)]]

            if weight > 0:
                total_weight, (average_x, average_y) = unit_cell_averages[(ax, ay)]
                new_x = average_x + dx*weight
                new_y = average_y + dy*weight
                new_weight = total_weight + weight
                unit_cell_averages[(ax, ay)] = (new_weight, (new_x, new_y))

    xs = []
    ys = []
    us = []
    vs = []
    for (cx, cy), (weight, (dx, dy)) in unit_cell_averages.items():
        x, y = graphbuilder.get_unit_cell_cartesian(cx, cy)
        xs.append(x)
        ys.append(y)
        us.append(dx / weight)
        # Note the minus sign, this is because we are inverting the y axis
        # and quiver only inverts the start positions of arrows, not the relative end.
        vs.append(-dy / weight)

    pyplot.title("Dimer biases")
    pyplot.quiver(xs, ys, us, vs)
    pyplot.grid()
    pyplot.gca().invert_yaxis()
    pyplot.savefig(filename)
    pyplot.clf()

    # Calculate divergence along boundary
    min_x = min(cx for (cx, _), _ in unit_cell_averages.items())
    max_x = max(cx for (cx, _), _ in unit_cell_averages.items())
    min_y = min(cy for (_, cy), _ in unit_cell_averages.items())
    max_y = max(cy for (_, cy), _ in unit_cell_averages.items())

    divergence = 0.0
    for (cx, cy), (weight, (dx, dy)) in unit_cell_averages.items():
        if cx == min_x:
            divergence -= dx/weight
        elif cx == max_x:
            divergence += dx/weight
        if cy == min_y:
            divergence -= dy / weight
        elif cy == max_y:
            divergence += dy / weight
    return divergence


def draw_flippable_states(filename, graph, sample, front=True):
    def flippable_color_fn(edge_a, edge_b):
        color_a = color_on_orientation_fn(*edge_a)
        color_b = color_on_orientation_fn(*edge_b)
        if color_a == color_b:
            return color_a
        else:
            return "gray"

    svg = graphdrawing.make_dimer_svg(graph, sample, front=front, dimer_color_fn=color_on_orientation_fn,
                                      flippable_color_fn=flippable_color_fn)
    if svg:
        with open(filename, "w") as w:
            w.write(svg)


def run_experiment_sweep(base_directory, experiment_gen, plot_functions=None):
    configs = None
    if not plot_functions:
        plot_functions = []
    pickle_path = os.path.join(base_directory, "configs.pickle")
    if os.path.exists(base_directory):
        if os.path.exists(pickle_path):
            print("Attempting to load experiment configurations from disk")
            with open(pickle_path, "rb") as f:
                configs = pickle.load(f)
                print("\tDone!")
    else:
        os.makedirs(base_directory)
    if configs is None:
        configs = [config for config in experiment_gen]
        print("Attempting to save experiment configurations to disk")
        with open(pickle_path, "wb") as w:
            pickle.dump(configs, w)
            print("\tDone!")

    scalars = collections.defaultdict(list)
    for i, experiment in enumerate(configs):
        experiment.run_or_load_experiment()
        experiment.add_default_analyzers()
        results = experiment.analyze()
        for k, v in results.items():
            scalars[k].append(v)
    for k, vs in scalars.items():
        with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
            for i, v in enumerate(vs):
                w.write("{}\t{}\n".format(i, v))

        pyplot.plot(vs, 'x--')
        pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
        pyplot.clf()

    for plot_fn in plot_functions:
        try:
            plot_fn(scalars)
        except Exception as e:
            print("Could not plot using {}".format(plot_fn))
            print(e)


def monte_carlo_sampler_fn():
    return monte_carlo_simulator.MonteCarloSampler()


def dwave_sampler_fn():
    return DWaveSampler()


def staggered_sampler_fn():
    return MockSampler()


if __name__ == "__main__":
    experiment_name = "data/tmp/test2"

    def experiment_gen(base_dir):
        n = 1
        for i in range(1, n+1):
            print("Building experiment {}".format(i))
            h = 0.0  # float(i) / n
            j = float(i) / n
            experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            print("\tUsing directory: {}".format(experiment_dir))
            config = ExperimentConfig(experiment_dir, monte_carlo_sampler_fn, h=h, j=j)
            config.num_reads = 1
            config.auto_scale = False
            config.build_graph(min_x=7, max_x=15, min_y=0, max_y=8, build_kwargs={'ideal_periodic_boundaries': True})
            yield config


    def defect_plot(scalars):
        inv_j = scalars['inv_j']
        defects = scalars['defect_count']
        defects_stdv = scalars['defect_stdv']
        hs = scalars['h']

        pyplot.errorbar(inv_j, defects, yerr=defects_stdv)
        pyplot.xlabel('1/J')
        pyplot.ylabel('Number of defects')
        pyplot.savefig(os.path.join(experiment_name, 'defects_vs_inv_j.svg'))
        pyplot.clf()

        pyplot.errorbar(hs, defects, yerr=defects_stdv)
        pyplot.xlabel('H')
        pyplot.ylabel('Number of defects')
        pyplot.savefig(os.path.join(experiment_name, 'defects_vs_hs.svg'))
        pyplot.clf()


    def flippable_plot(scalars):
        inv_j = scalars['inv_j']
        flippable_count = scalars['flippable_count']
        flippable_stdv = scalars['flippable_stdv']
        hs = scalars['h']

        pyplot.errorbar(inv_j, flippable_count, yerr=flippable_stdv)
        pyplot.xlabel('1/J')
        pyplot.ylabel('Number of flippable plaquettes')
        pyplot.savefig(os.path.join(experiment_name, 'flippable_vs_inv_j.svg'))
        pyplot.clf()

        pyplot.errorbar(hs, flippable_count, yerr=flippable_stdv)
        pyplot.xlabel('H')
        pyplot.ylabel('Number of flippable plaquettes')
        pyplot.savefig(os.path.join(experiment_name, 'flippable_vs_hs.svg'))
        pyplot.clf()

    def unit_cell_divergence_plot(scalars):
        inv_j = scalars['inv_j']
        divergence = scalars['unit_cell_divergence']
        hs = scalars['h']

        pyplot.plot(inv_j, divergence)
        pyplot.xlabel('1/J')
        pyplot.ylabel('Boundary Divergence')
        pyplot.savefig(os.path.join(experiment_name, 'divergence_vs_inv_j.svg'))
        pyplot.clf()

    run_experiment_sweep(experiment_name, experiment_gen(experiment_name),
                         plot_functions=[defect_plot, flippable_plot, unit_cell_divergence_plot])
