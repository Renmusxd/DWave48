import numpy
import graphbuilder
import collections


class GraphAnalyzer:
    """Things about the graph using data from the samples."""
    def __init__(self, graph, samples):
        self.graph = graph
        self.samples = samples

        # Memoization for later
        self.sorted_edges = None
        # From get_flat_samples
        self.var_map = None
        self.var_mat = None

        # from get_correlation_matrix
        self.variable_correlations = None

        # from calculate_correlation_function
        self.variable_distance_correlations = None
        self.variable_distance_stdv = None

        # from get_dimer_matrix
        self.dimer_matrix = None

        # from get_dimer_correlations
        self.dimer_correlation = None

        # from get_dimer_vertex_counts
        self.vertex_counts = None
        self.dimer_to_flippable_matrix = None
        self.flippable_squares = None
        self.dimer_distance_correlations = None
        self.dimer_distance_stdv = None

    def get_flat_samples(self):
        if self.var_map is None or self.var_mat is None:
            var_map, var_mat = flatten_dicts(self.samples)
            self.var_map = var_map
            self.var_mat = var_mat
        return self.var_map, self.var_mat

    def get_correlation_matrix(self):
        if self.variable_correlations is None:
            _, var_mat = self.get_flat_samples()
            self.variable_correlations = calculate_correlation_matrix(var_mat)
        return self.variable_correlations

    def calculate_correlation_function(self):
        """
        Calculate correlations as a function of distance for each variable.
        :return: NxN matrix of variable correlations, NxD matrix of variable distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        var_corr = self.get_correlation_matrix()
        if self.variable_distance_correlations is None or self.variable_distance_stdv is None:
            # max distance (protected from infinity)
            max_dist = min(numpy.max(self.graph.vertex_distances), len(self.graph.edges))
            distance_corrs = numpy.zeros((len(self.graph.all_vars), max_dist + 1))
            distance_stdv = numpy.zeros((len(self.graph.all_vars), max_dist + 1))
            # Need to find a numpy-ish way to do this stuff.
            for i, v in enumerate(self.graph.all_vars):
                totals = numpy.zeros(max_dist + 1)
                for j in range(len(self.graph.all_vars)):
                    corr = var_corr[i, j]
                    d = self.graph.vertex_distances[i, j]
                    if d <= max_dist:
                        totals[d] += 1
                        distance_corrs[i, d] += corr
                distance_corrs[i, :] = distance_corrs[i, :] / totals
                for j in range(len(self.graph.all_vars)):
                    corr = var_corr[i, j]
                    d = self.graph.vertex_distances[i, j]
                    if d <= max_dist:
                        distance_stdv[i, d] += (corr - distance_corrs[i, d]) ** 2
                distance_stdv[i, :] = numpy.sqrt(distance_stdv[i, :] / totals)
            self.variable_distance_correlations = distance_corrs
            self.variable_distance_stdv = distance_stdv
        return var_corr, self.variable_distance_correlations, self.variable_distance_stdv, self.graph.all_vars

    def get_dimer_matrix(self):
        """Get matrix of dimers for each sample."""
        if self.dimer_matrix is None:
            # Get flat versions of variables, values, and make lookup tables.
            var_map, var_mat = self.get_flat_samples()
            var_lookup = {v: k for k, v in enumerate(var_map)}
            edge_list = numpy.asarray([(var_lookup[a], var_lookup[b]) for a, b in self.graph.sorted_edges])
            edge_values = numpy.asarray([self.graph.edges[edge] for edge in self.graph.sorted_edges])
            a_values = var_mat[edge_list[:, 0], :]
            b_values = var_mat[edge_list[:, 1], :]

            # Gives +1 if variables are equal, -1 if not
            spin_product = a_values * b_values
            # Gives +1 if edge is positive, -1 if not.
            edge_signs = numpy.expand_dims(numpy.sign(edge_values), -1)

            # If edge is positive, and product is negative (unequal), then satisfied.
            # So satisfied edges are negative, broken are positive
            self.dimer_matrix = spin_product * edge_signs
        return self.dimer_matrix

    def get_dimer_correlations(self):
        """
        Get the correlation of broken bonds (dimers).
        :param edges: map {(int, int): float} (edges and their couplings)
        :param samples: list of maps {int: float} for each sample
        :return: DxD matrix of correlations.
        """
        # Get dimers per sample
        edges_broken = self.get_dimer_matrix()
        if self.dimer_correlation is None:
            # Now get correlations
            self.dimer_correlation = calculate_correlation_matrix(edges_broken)
        return self.dimer_correlation

    def calculate_dimer_correlation_function(self):
        """
        Calculate correlations as a function of distance for each dimer.
        :return: NxN matrix of dimer correlations, NxD matrix of dimer distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        # all_dimer_pairs maps the index in dimer_distances to the index into self.sorted_edges
        dimer_corr = self.get_dimer_correlations()
        if self.dimer_distance_correlations is None or self.dimer_distance_stdv is None:
            # max distance (protected from infinity)
            max_dist = min(numpy.max(self.graph.dimer_distance_mat), self.graph.dimer_distance_mat.shape[0])
            # Matrices to store values
            distance_corrs = numpy.zeros((len(self.graph.all_dimer_pairs), max_dist + 1))
            distance_stdv = numpy.zeros((len(self.graph.all_dimer_pairs), max_dist + 1))
            # Need to find a numpy-ish way to do this stuff.
            for i, v in enumerate(self.graph.all_dimer_pairs):
                totals = numpy.zeros(max_dist + 1)
                for j in range(len(self.graph.all_dimer_pairs)):
                    corr = dimer_corr[i, j]
                    d = self.graph.dimer_distance_mat[i, j]
                    if d <= max_dist:
                        totals[d] += 1
                        distance_corrs[i, d] += corr
                distance_corrs[i, :] = distance_corrs[i, :] / totals
                for j in range(len(self.graph.all_dimer_pairs)):
                    corr = dimer_corr[i, j]
                    d = self.graph.dimer_distance_mat[i, j]
                    if d <= max_dist:
                        distance_stdv[i, d] += (corr - distance_corrs[i, d]) ** 2
                distance_stdv[i, :] = numpy.sqrt(distance_stdv[i, :] / totals)
            self.dimer_distance_correlations = distance_corrs
            self.dimer_distance_stdv = distance_stdv
        return dimer_corr, self.dimer_distance_correlations, self.dimer_distance_stdv, self.graph.all_dimer_pairs

    def get_dimer_vertex_counts(self):
        if self.vertex_counts is None:
            # Get dimers per sample
            edges_broken = self.get_dimer_matrix()
            edge_to_vertex_matrix = self.graph.edge_to_vertex_matrix
            self.vertex_counts = numpy.matmul(edge_to_vertex_matrix.T, edges_broken == 1)
        return self.vertex_counts

    def get_flippable_squares_list(self):
        if self.flippable_squares is None:
            # Each cell-cell connection gives a flippable square.
            # Label each by (cell)-(cell)
            unit_cells = set(self.graph.unit_cells)
            connections = []
            for (cx, cy, front) in unit_cells:
                # Check cx+1 and cy+1, the [cu-1] will be checked by the other ones
                oa = (cx + 1, cy, front)
                ob = (cx, cy + 1, front)
                for other in [oa, ob]:
                    if other in unit_cells:
                        connections.append(((cx, cy, front), other))
            self.flippable_squares = list(sorted(connections))
        return self.flippable_squares

    def get_dimer_to_flippable_squares_matrix(self):
        if self.dimer_to_flippable_matrix is None:
            flippable_squares = self.get_flippable_squares_list()
            connection_indices = {k: i for i, k in enumerate(flippable_squares)}
            self.dimer_to_flippable_matrix = numpy.zeros((len(self.graph.edges), len(flippable_squares)),
                                                         dtype=numpy.int8)
            for i, (vara, varb) in enumerate(self.graph.sorted_edges):
                ax, ay, arel = graphbuilder.get_var_traits(vara, vars_per_cell=self.graph.vars_per_cell,
                                                           unit_cells_per_row=self.graph.unit_cells_per_row)
                bx, by, brel = graphbuilder.get_var_traits(varb, vars_per_cell=self.graph.vars_per_cell,
                                                           unit_cells_per_row=self.graph.unit_cells_per_row)
                # Only consider two variables in same cell.
                if ax != bx or ay != by:
                    continue
                cx, cy = ax, ay
                # TODO fix this whenever periodic is really working.
                front = graphbuilder.is_front(vara, vars_per_cell=self.graph.vars_per_cell)

                dax, day = graphbuilder.calculate_variable_direction(vara, vars_per_cell=self.graph.vars_per_cell,
                                                                     unit_cells_per_row=self.graph.unit_cells_per_row)
                dbx, dby = graphbuilder.calculate_variable_direction(varb, vars_per_cell=self.graph.vars_per_cell,
                                                                     unit_cells_per_row=self.graph.unit_cells_per_row)
                for dx, dy in [(dax, day), (dbx, dby)]:
                    ox, oy = cx+dx, cy+dy
                    connection = tuple(sorted(((cx, cy, front), (ox, oy, front))))
                    if connection in connection_indices:
                        j = connection_indices[connection]
                        self.dimer_to_flippable_matrix[i, j] = 1
        return self.dimer_to_flippable_matrix

    def get_flippable_squares(self):
        dimers = self.get_dimer_matrix()
        dimer_to_flippable = self.get_dimer_to_flippable_squares_matrix()
        num_dimers_adjacent_to_flippables = numpy.matmul(dimer_to_flippable.T, dimers==1)
        flippable_states = num_dimers_adjacent_to_flippables == 2
        return flippable_states


def flatten_dicts(samples):
    """
    Flattens a list of dicts into a mapping and a list
    :param samples: Mapping from variable index to value
    :return:
    """
    sample = samples[0]
    base_sample = {k: [v] for k, v in sample.items()}
    for sample in samples[1:]:
        for k, v in sample.items():
            base_sample[k].append(v)
    keys = numpy.asarray(list(sorted(base_sample)))
    values = numpy.asarray([base_sample[k] for k in keys])
    return keys, values


def calculate_correlation_matrix(variable_values):
    """
    Calculates the cosine similarity between each set of variables over time.
    :param variable_values: nxd matrix of n variables and their values across d runs.
    :return: nxn matrix of the cosine similarities
    """
    variable_values = variable_values * 1.0
    m = numpy.matmul(variable_values, variable_values.T)
    v_sqr = numpy.square(variable_values)
    v_norm_sqr = numpy.sum(v_sqr, -1)
    v_norms = numpy.sqrt(v_norm_sqr)
    norm_mat = numpy.outer(v_norms, v_norms)
    norm_m = m / norm_mat
    return norm_m


def edge_is_satisfied(variable_values, edges, vara, varb):
    """Returns true if edge is in minimum energy state."""
    j_val = edges[(min(vara, varb), max(vara, varb))]
    if j_val > 0:
        return variable_values[vara] != variable_values[varb]
    else:
        return variable_values[vara] == variable_values[varb]