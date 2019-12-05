from dwave.system.samplers import DWaveSampler
import graphbuilder
import graphanalysis
import graphdrawing
import pickle
import os
import collections
from matplotlib import pyplot
import numpy


class ExperimentConfig:
    def __init__(self, base_dir, machine_temp=14.5e-3):
        self.base_dir = base_dir
        self.graph = None
        self.hs = {}
        self.num_reads = 0
        self.auto_scale = True
        self.data = None
        self.machine_temp = machine_temp
        self.effective_temp = self.machine_temp

    def build_graph(self, max_x=8, max_y=16, min_x=0, min_y=0, j=1.0, hs_override=None):
        graph = graphbuilder.Graph(j=j)
        graph.add_cells([
            (x, y)
            for x in range(min_x, max_x)
            for y in range(min_y, max_y)
        ])
        graph.connect_all()
        self.hs, self.graph = graph.build()
        self.effective_temp = self.machine_temp / j
        if hs_override is not None:
            self.hs.update(hs_override)

    def maybe_load_self(self, filepath):
        if os.path.exists(filepath):
            print("\tLoading self... ", end='')
            try:
                with open(filepath, "rb") as f:
                    config = pickle.load(f)
                self.override(config)
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

            response = DWaveSampler().sample_ising(self.hs, self.graph,
                                                   num_reads=self.num_reads, auto_scale=self.auto_scale)
            self.data = [({k: sample[k] for k in sample}, energy, num_occurences) for sample, energy, num_occurences in
                         response.data()]
            print("done!")
            self.save_self(filepath)

    def analyze(self):
        print("\tRunning analysis...")
        def color_by_sign(var_a, var_b):
            if self.graph[(var_a, var_b)] < 0:
                return "rgb(0,0,0)"
            else:
                return "rgb(255,0,0)"

        print("\tMaking lattice svgs")
        svg = graphdrawing.make_edges_svg(list(self.graph), color_fn=color_by_sign, front=True)
        if svg:
            with open(os.path.join(self.base_dir, "front_lattice.svg"), "w") as w:
                w.write(svg)
        svg = graphdrawing.make_edges_svg(list(self.graph), color_fn=color_by_sign, front=False)
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
            draw_dimers(os.path.join(self.base_dir, "front_min_energy_dimers.svg"), self.graph, sample,
                        front=True, color_on_orientation=False)
            draw_dimers(os.path.join(self.base_dir, "front_min_energy_dimers_color.svg"), self.graph, sample,
                        front=True, color_on_orientation=True)

            draw_dimers(os.path.join(self.base_dir, "rear_min_energy_dimers.svg"), self.graph, sample,
                        front=False, color_on_orientation=False)
            draw_dimers(os.path.join(self.base_dir, "rear_min_energy_dimers_color.svg"), self.graph, sample,
                        front=False, color_on_orientation=False)

            # Now plot the correlation matrix for the variables, and the correlation as a function of distance.

            print("\tCalculating variable correlations")
            var_corr, distance_corr, distance_stdv, all_vars = graph_analyzer.calculate_correlation_function()
            distance_corr = numpy.nan_to_num(distance_corr)
            distance_stdv = numpy.nan_to_num(distance_stdv)
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

            # Similarly, get the correlation plot for the dimers
            print("\tCalculating dimer correlations")
            _, dimer_corrs = graph_analyzer.get_dimer_correlations()
            pyplot.imshow(dimer_corrs, interpolation='nearest')
            pyplot.colorbar()
            pyplot.savefig(os.path.join(self.base_dir, "dimer_correlations.svg"))
            pyplot.clf()

            # Get the average dimer occupations
            print("\tDrawing dimer occupations")
            average_dimers, stdv_dimers = draw_occupations(os.path.join(self.base_dir, "dimer_occupation_graph.svg"),
                                                           self.graph, graph_analyzer)
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

            # Count the defects
            print("\tCounting defects")
            defects = graph_analyzer.get_dimer_vertex_counts() > 1
            total_defects_per_sample = numpy.sum(defects, 0)
            average_defects = numpy.mean(total_defects_per_sample, -1)
            stdv_defects = numpy.sqrt(numpy.var(total_defects_per_sample, -1))

            defects = (average_defects, stdv_defects)
            print("\tDone!")
            return ExperimentResults(self.base_dir, defects, effective_temp=self.effective_temp)


class ExperimentResults:
    def __init__(self, filepath, defects, effective_temp=None):
        self.filepath = filepath
        self.defects = defects
        self.effective_temp = effective_temp

    def get_named_scalars(self):
        return {
            "defect_count": self.defects[0],
            "defect_stdv": self.defects[1],
            "effective_temp": self.effective_temp
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


def draw_dimers(filename, graph, sample, front=True, color_on_orientation=True):

    def color_on_orientation_fn(var_a, var_b):
        dx_a, dy_a = graphanalysis.calculate_variable_direction(var_a)
        dx_b, dy_b = graphanalysis.calculate_variable_direction(var_b)
        # Vertical and horizontal bonds are green (and rare)
        if dx_a == -dx_b or dy_a == -dy_b:
            return "green"
        if dx_a == dy_b and dy_a == dx_b:
            return "red"
        if dx_a == -dy_b and dy_a == -dx_b:
            return "blue"
        return "purple"

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


def draw_occupations(filename, graph, graph_analyzer, front=True):
    edge_lookup, dimer_matrix = graph_analyzer.get_dimer_matrix()
    average_dimers = numpy.mean(dimer_matrix == 1, -1)
    stdv_dimers = numpy.var(dimer_matrix == 1, -1)

    # Basic color scheme
    def dimer_color_fn(var_a, var_b):
        edge = (min(var_a, var_b), max(var_a, var_b))
        edge_indx = edge_lookup[edge]
        average_values = average_dimers[edge_indx]
        red_color = int(average_values*256)
        return "rgb({},0,0)".format(red_color)

    svg = graphdrawing.make_dimer_contents(graph, front=front, dimer_color_fn=dimer_color_fn)
    svg = graphdrawing.wrap_with_svg(svg)
    if svg:
        with open(filename, "w") as w:
            w.write(svg)

    return average_dimers, stdv_dimers


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


if __name__ == "__main__":
    experiment_name = "data/j_sweep"

    def experiment_gen(base_dir):
        n = 10
        for i in range(1, n):
            print("Running experiment {}".format(i))
            experiment_dir = os.path.join(base_dir, "experiment_{}".format(i))
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            print("\tUsing directory: {}".format(experiment_dir))
            config = ExperimentConfig(experiment_dir)
            config.num_reads = 10000
            config.auto_scale = False
            config.build_graph(j=float(i)/n)
            yield config

    def defect_plot(scalars):
        effective_temperatures = scalars['effective_temp']
        defects = scalars['defect_count']
        defects_stdv = scalars['defect_stdv']
        pyplot.errorbar(effective_temperatures, defects, yerr=defects_stdv)
        pyplot.xlabel('Effective temperature (K)')
        pyplot.ylabel('Number of defects')

        pyplot.savefig(os.path.join(experiment_name, 'defects_vs_temp.svg'))
        pyplot.clf()

    run_experiment_sweep(experiment_name, experiment_gen(experiment_name),
                         plot_functions=[defect_plot])
