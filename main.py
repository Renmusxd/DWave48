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


class ExperimentConfig:
    def __init__(self, base_dir, sampler_fn, machine_temp=14.5e-3, h=0.0, j=1.0):
        self.base_dir = base_dir
        self.sampler_fn = sampler_fn
        self.graph = None
        self.hs = {}
        self.num_reads = 0
        self.auto_scale = True
        self.data = None
        self.machine_temp = machine_temp
        self.h = h
        self.j = j

    def build_graph(self, max_x=8, max_y=16, min_x=0, min_y=0, hs_override=None):
        gb = graphbuilder.GraphBuilder(j=self.j)
        gb.add_cells([
            (x, y)
            for x in range(min_x, max_x)
            for y in range(min_y, max_y)
        ])
        gb.connect_all()
        self.graph = gb.build(h=self.h)
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
        self.data = config.data
        self.graph = config.graph

    def run_or_load_experiment(self):
        filepath = os.path.join(self.base_dir, "config.pickle")
        if not self.maybe_load_self(filepath):
            if self.graph is None:
                raise Exception("Graph not yet built")
            print("\tRunning on dwave... ", end='')
            response = self.sampler_fn().sample_ising(self.hs, self.graph.edges,
                                                      num_reads=self.num_reads,
                                                      auto_scale=self.auto_scale)

            self.data = [({k: sample[k] for k in sample}, energy, num_occurences) for sample, energy, num_occurences in
                         response.data()]
            print("done!")
            self.save_self(filepath)

    def analyze(self):
        print("\tRunning analysis...")

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

        if self.data:
            samples = [sample for sample, _, __ in self.data]
            graph_analyzer = graphanalysis.GraphAnalyzer(self.graph, samples)

            # First plot the energies that we found
            energies = [energy for _, energy, __ in self.data]
            num_occurrences = [num_occurrences for _, __, num_occurrences in self.data]

            print("\tMaking energy histogram")
            pyplot.hist(energies, weights=num_occurrences)
            pyplot.savefig(os.path.join(self.base_dir, "energies.svg"))
            pyplot.clf()

            # Then plot the dimers for one of the ground states (or lowest E we found anyway).
            print("\tDrawing dimer svgs")
            lowest_e = numpy.argmin(energies)
            sample, energy, num_occurrences = self.data[lowest_e]
            draw_dimers(os.path.join(self.base_dir, "front_min_energy_dimers.svg"), self.graph.edges, sample,
                        front=True, color_on_orientation=False)
            draw_dimers(os.path.join(self.base_dir, "front_min_energy_dimers_color.svg"), self.graph.edges, sample,
                        front=True, color_on_orientation=True)

            draw_dimers(os.path.join(self.base_dir, "rear_min_energy_dimers.svg"), self.graph.edges, sample,
                        front=False, color_on_orientation=False)
            draw_dimers(os.path.join(self.base_dir, "rear_min_energy_dimers_color.svg"), self.graph.edges, sample,
                        front=False, color_on_orientation=False)

            # Now plot the correlation matrix for the variables, and the correlation as a function of distance.
            print("\tCalculating variable correlations")
            var_corr, distance_corr, _, __ = graph_analyzer.calculate_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            pyplot.imshow(var_corr, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(self.base_dir, "variable_correlations.svg"))
            pyplot.clf()

            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Distance (in # edges)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(self.base_dir, "correlation_distance.svg"))
            pyplot.clf()

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
            pyplot.savefig(os.path.join(self.base_dir, "correlation_euclidean_distance.svg"))
            pyplot.clf()

            # Similarly, get the correlation plot for the dimers
            print("\tCalculating dimer correlations")
            dimer_corrs, distance_corr, _, __ = graph_analyzer.calculate_dimer_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            pyplot.imshow(dimer_corrs, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(self.base_dir, "dimer_correlations.svg"))
            pyplot.clf()

            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (in # edges in dimer-dual)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(self.base_dir, "dimer_correlation_distance.svg"))
            pyplot.clf()

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
            pyplot.savefig(os.path.join(self.base_dir, "dimer_correlation_euclidean_distance.svg"))
            pyplot.clf()

            # Similarly, get the correlation plot for the dimers
            print("\tCalculating diagonal dimer correlations")
            dimer_corrs, distance_corr, _, __ = graph_analyzer.calculate_diagonal_dimer_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            pyplot.imshow(dimer_corrs, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(self.base_dir, "diagonal_dimer_correlations.svg"))
            pyplot.clf()

            # Now the distance part, with error bars.
            print("\tCalculating distance correlations")
            average_corrs = numpy.mean(distance_corr, 0)
            stdv_corrs = numpy.sqrt(numpy.var(distance_corr, 0))
            xs = numpy.arange(average_corrs.shape[0])
            pyplot.errorbar(xs, average_corrs, yerr=stdv_corrs, label="Average")
            pyplot.legend()
            pyplot.grid()
            pyplot.xlabel("Dimer distance (in # edges in dimer-dual)")
            pyplot.ylabel("Correlation")
            pyplot.savefig(os.path.join(self.base_dir, "diagonal_dimer_correlation_distance.svg"))
            pyplot.clf()

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
            pyplot.savefig(os.path.join(self.base_dir, "diagonal_dimer_correlation_euclidean_distance.svg"))
            pyplot.clf()

            # Get the average dimer occupations
            print("\tDrawing dimer occupations")
            average_dimers, stdv_dimers = draw_occupations(os.path.join(self.base_dir, "dimer_occupation_graph.svg"),
                                                           self.graph.edges, graph_analyzer)
            average_dimers, stdv_dimers = draw_occupations(os.path.join(self.base_dir, "dimer_occupation_graph_scaled.svg"),
                                                           self.graph.edges, graph_analyzer, scale=True)
            draw_average_unit_cell_directions(os.path.join(self.base_dir, "dimer_biases.svg"), graph_analyzer)

            # Sort them
            average_dimers, stdv_dimers = zip(*sorted(zip(average_dimers, stdv_dimers), key=lambda x: x[0]))
            average_dimers = numpy.asarray(average_dimers)
            stdv_dimers = numpy.asarray(stdv_dimers)
            xs = numpy.arange(len(average_dimers))
            pyplot.plot(xs, average_dimers)
            pyplot.plot(xs, average_dimers + stdv_dimers, 'r--')
            pyplot.plot(xs, average_dimers - stdv_dimers, 'r--')
            pyplot.savefig(os.path.join(self.base_dir, "dimer_occupation_plot.svg"))
            pyplot.clf()

            pyplot.hist(average_dimers)
            pyplot.xlabel("Dimer occupation frequencies")
            pyplot.ylabel("Counts")
            pyplot.savefig(os.path.join(self.base_dir, "dimer_occupation_hist.svg"))
            pyplot.clf()

            # Get flippable plaquettes
            print("\tDrawing flippable states")
            draw_flippable_states(os.path.join(self.base_dir, "dimer_flippable_plot.svg"), self.graph.edges, sample)

            print("\tCalculating flippable count")
            flippable_squares = graph_analyzer.get_flippable_squares()
            flippable_count = numpy.mean(numpy.sum(flippable_squares, 0))
            flippable_stdv = numpy.sqrt(numpy.var(numpy.sum(flippable_squares, 0)))

            # Count the defects
            print("\tCounting defects")
            defects = graph_analyzer.get_defects()
            total_defects_per_sample = numpy.sum(defects, 0)
            average_defects = numpy.mean(total_defects_per_sample, -1)
            stdv_defects = numpy.sqrt(numpy.var(total_defects_per_sample, -1))

            # Get defect correlations
            print("\tDefect correlations")
            defects_corr, defects_corr_function, _, _ = graph_analyzer.calculate_euclidean_defect_correlation_function()
            defects_corr = numpy.nan_to_num(defects_corr)
            pyplot.imshow(defects_corr, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(self.base_dir, "defect_correlations.svg"))
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
            pyplot.savefig(os.path.join(self.base_dir, "defect_correlation_distance.svg"))
            pyplot.clf()

            defects = (average_defects, stdv_defects)
            flippables = (flippable_count, flippable_stdv)
            print("\tDone!")
            return ExperimentResults(self.base_dir, defects, flippables, self.j, self.h)


class ExperimentResults:
    def __init__(self, filepath, defects, flippables, j, h):
        self.filepath = filepath
        self.defects = defects
        self.flippables = flippables
        self.j = j
        self.h = h

    def get_named_scalars(self):
        return {
            "defect_count": self.defects[0],
            "defect_stdv": self.defects[1],
            "flippable_count": self.flippables[0],
            "flippable_stdv": self.flippables[1],
            "j": self.j,
            "inv_j": 1.0 / self.j,
            "h": self.h
        }


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


def draw_dimers(filename, graph, sample, front=True, color_on_orientation=True):
    if not color_on_orientation:
        # Basic color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front)
        if svg:
            with open(filename, "w") as w:
                w.write(svg)
    else:
        # Orientation color scheme
        svg = graphdrawing.make_dimer_svg(graph, sample, front=front, dimer_color_fn=color_on_orientation_fn)
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
    # TODO debug this thing
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
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    scalars = collections.defaultdict(list)
    for i, experiment in enumerate(experiment_gen):
        experiment.run_or_load_experiment()
        results = experiment.analyze()
        for k, v in results.get_named_scalars().items():
            scalars[k].append(v)
    for k, vs in scalars.items():
        with open(os.path.join(base_directory, "{}.txt".format(k)), "w") as w:
            for i, v in enumerate(vs):
                w.write("{}\t{}\n".format(i, v))

        pyplot.plot(vs, 'x--')
        pyplot.savefig(os.path.join(base_directory, "{}.svg".format(k)))
        pyplot.clf()

    if plot_functions:
        for plot_fn in plot_functions:
            plot_fn(scalars)


def monte_carlo_sampler_fn():
    return monte_carlo_simulator.MonteCarloSampler()


def dwave_sampler_fn():
    return DWaveSampler()


if __name__ == "__main__":
    experiment_name = "data/j_sweep"

    def experiment_gen(base_dir):
        n = 10
        for i in range(1, n):
            print("Running experiment {}".format(i))
            h = 0.0  # float(i) / n
            j = float(i) / n
            experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            print("\tUsing directory: {}".format(experiment_dir))
            config = ExperimentConfig(experiment_dir, dwave_sampler_fn, h=h, j=j)
            config.num_reads = 10000
            config.auto_scale = False
            config.build_graph()
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


    run_experiment_sweep(experiment_name, experiment_gen(experiment_name),
                         plot_functions=[defect_plot, flippable_plot])
