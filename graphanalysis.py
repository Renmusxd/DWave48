import numpy
import graphbuilder
import collections


class GraphAnalyzer:
    """Things about the graph using data from the samples."""

    def __init__(self, graph, samples):
        self.graph = graph
        self.samples = samples

        # From get_flat_samples
        self.var_map = None
        self.var_mat = None

        # from get_correlation_matrix
        self.variable_correlations = None

        # from calculate_correlation_function
        self.variable_distance_correlations = None
        self.variable_distance_stdv = None
        self.variable_euclidean_distance_correlations = None
        self.variable_euclidean_distance_stdv = None
        self.dimer_euclidean_distance_correlations = None
        self.dimer_euclidean_distance_stdv = None

        # from get_dimer_matrix
        self.dimer_matrix = None
        # and diagonal version
        self.diagonal_dimer_matrix = None

        # from get_dimer_correlations
        self.dimer_correlation = None
        # and diagonal version
        self.diagonal_dimer_correlation = None

        # from get_dimer_vertex_counts
        self.vertex_counts = None
        self.dimer_to_flippable_matrix = None
        self.flippable_squares = None
        self.dimer_distance_correlations = None
        self.dimer_distance_stdv = None

        # diagonals
        self.diagonal_dimer_distance_correlations = None
        self.diagonal_dimer_distance_stdv = None

        # from get_defect_correlations
        self.defect_correlations = None

        self.defect_euclidean_distance_correlations = None
        self.defect_euclidean_distance_stdv = None

        # diagonals
        self.diagonal_dimer_euclidean_distance_correlations = None
        self.diagonal_dimer_euclidean_distance_stdv = None

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
            distance_corrs, distance_stdv = average_by_distance(self.graph.vertex_distances, var_corr)
            self.variable_distance_correlations = distance_corrs
            self.variable_distance_stdv = distance_stdv
        return var_corr, self.variable_distance_correlations, self.variable_distance_stdv, self.graph.all_vars

    def calculate_euclidean_correlation_function(self):
        """
        Calculate correlations as a function of euclidean distance for each variable.
        :return: NxN matrix of variable correlations, NxD matrix of variable distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        var_corr = self.get_correlation_matrix()
        if self.variable_euclidean_distance_correlations is None or self.variable_euclidean_distance_stdv is None:
            distance_corrs, distance_stdv = average_by_distance(self.graph.vertex_euclidean_distances, var_corr)
            self.variable_euclidean_distance_correlations = distance_corrs
            self.variable_euclidean_distance_stdv = distance_stdv
        return var_corr, self.variable_euclidean_distance_correlations, self.variable_euclidean_distance_stdv, self.graph.all_vars

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
            distance_corrs, distance_stdv = average_by_distance(self.graph.dimer_distance_mat, dimer_corr)
            self.dimer_distance_correlations = distance_corrs
            self.dimer_distance_stdv = distance_stdv
        return dimer_corr, self.dimer_distance_correlations, self.dimer_distance_stdv, self.graph.all_dimer_pairs

    def calculate_euclidean_dimer_correlation_function(self):
        """
        Calculate correlations as a function of distance for each dimer.
        :return: NxN matrix of dimer correlations, NxD matrix of dimer distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        dimer_corr = self.get_dimer_correlations()
        if self.dimer_euclidean_distance_correlations is None or self.dimer_euclidean_distance_stdv is None:
            distance_corrs, distance_stdv = average_by_distance(self.graph.dimer_euclidean_distances, dimer_corr)
            self.dimer_euclidean_distance_correlations = distance_corrs
            self.dimer_euclidean_distance_stdv = distance_stdv
        return dimer_corr, self.dimer_euclidean_distance_correlations, self.dimer_euclidean_distance_stdv, self.graph.all_vars

    def get_diagonal_dimer_mask(self):
        """List of True/False for each edge whether it is diagonal or not."""

        def is_diagonal(vara, varb):
            ax, ay, _ = graphbuilder.get_var_traits(vara, self.graph.vars_per_cell, self.graph.unit_cells_per_row)
            bx, by, _ = graphbuilder.get_var_traits(varb, self.graph.vars_per_cell, self.graph.unit_cells_per_row)
            return ax == bx and ay == by

        return [is_diagonal(*edge) for edge in self.graph.sorted_edges]

    def get_nesw_dimer_mask(self):
        return [get_variable_orientation(*edge) == -1 for edge in self.graph.sorted_edges]

    def get_nwse_dimer_mask(self):
        return [get_variable_orientation(*edge) == 1 for edge in self.graph.sorted_edges]

    def get_diagonal_dimer_matrix(self):
        """Same as get_dimer_matrix but only dimers which are across edges within each 4-cell."""
        if self.diagonal_dimer_matrix is None:
            diagonal_mask = self.get_diagonal_dimer_mask()
            dimers = self.get_dimer_matrix()
            self.diagonal_dimer_matrix = dimers[diagonal_mask, :]
        return self.diagonal_dimer_matrix

    def get_nesw_dimer_matrix(self):
        """Same as get_dimer_matrix but only dimers which are ne-sw edges within each 4-cell."""
        if self.nesw_dimer_matrix is None:
            mask = self.get_nesw_dimer_mask()
            dimers = self.get_dimer_matrix()
            self.nesw_dimer_matrix = dimers[mask, :]
        return self.nesw_dimer_matrix

    def get_nwse_dimer_matrix(self):
        """Same as get_dimer_matrix but only dimers which are nw-se edges within each 4-cell."""
        if self.nwse_dimer_matrix is None:
            mask = self.get_nwse_dimer_mask()
            dimers = self.get_dimer_matrix()
            self.nwse_dimer_matrix = dimers[mask, :]
        return self.nwse_dimer_matrix

    def get_diagonal_dimer_correlations(self):
        """
        Get the correlation of broken bonds (dimers).
        :param edges: map {(int, int): float} (edges and their couplings)
        :param samples: list of maps {int: float} for each sample
        :return: DxD matrix of correlations.
        """
        # Get dimers per sample
        edges_broken = self.get_diagonal_dimer_matrix()
        if self.diagonal_dimer_correlation is None:
            # Now get correlations
            self.diagonal_dimer_correlation = calculate_correlation_matrix(edges_broken)
        return self.diagonal_dimer_correlation

    def calculate_diagonal_dimer_correlation_function(self):
        """
        Calculate correlations as a function of distance for each diagonal dimer.
        :return: NxN matrix of dimer correlations, NxD matrix of dimer distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        # all_dimer_pairs maps the index in dimer_distances to the index into self.sorted_edges
        dimer_corr = self.get_diagonal_dimer_correlations()
        if self.diagonal_dimer_distance_correlations is None or self.diagonal_dimer_distance_stdv is None:
            diagonal_mask = self.get_diagonal_dimer_mask()
            diagonal_distances = self.graph.dimer_distance_mat[numpy.ix_(diagonal_mask, diagonal_mask)]
            distance_corrs, distance_stdv = average_by_distance(diagonal_distances, dimer_corr)
            self.diagonal_dimer_distance_correlations = distance_corrs
            self.diagonal_dimer_distance_stdv = distance_stdv
        return dimer_corr, self.diagonal_dimer_distance_correlations, self.diagonal_dimer_distance_stdv, self.graph.all_dimer_pairs

    def calculate_euclidean_diagonal_dimer_correlation_function(self):
        """
        Calculate correlations as a function of distance for each diagonal dimer.
        :return: NxN matrix of dimer correlations, NxD matrix of dimer distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to variable index.
        """
        dimer_corr = self.get_diagonal_dimer_correlations()
        if self.diagonal_dimer_euclidean_distance_correlations is None or self.diagonal_dimer_euclidean_distance_stdv is None:
            diagonal_mask = self.get_diagonal_dimer_mask()
            diagonal_distances = self.graph.dimer_euclidean_distances[numpy.ix_(diagonal_mask, diagonal_mask)]
            distance_corrs, distance_stdv = average_by_distance(diagonal_distances, dimer_corr)
            self.diagonal_dimer_euclidean_distance_correlations = distance_corrs
            self.diagonal_dimer_euclidean_distance_stdv = distance_stdv
        return dimer_corr, self.diagonal_dimer_euclidean_distance_correlations, self.diagonal_dimer_euclidean_distance_stdv, self.graph.all_vars

    def calculate_oriented_dimer_correlation_function(self):
        """Get correlations between specific orientations of dimers."""
        nesw_mask = self.get_nesw_dimer_mask()
        nwse_mask = self.get_nwse_dimer_mask()
        correlations = self.get_dimer_correlations()
        masks = [(nesw_mask, nesw_mask), (nesw_mask, nwse_mask), (nwse_mask, nesw_mask), (nwse_mask, nwse_mask)]
        corrs = []
        stdvs = []
        for mask_a, mask_b in masks:
            sub_corrs = correlations[numpy.ix_(mask_a, mask_b)]
            sub_distances = self.graph.dimer_distance_mat[numpy.ix_(mask_a, mask_b)]
            sub_corrs, sub_stdv = average_by_distance(sub_distances, sub_corrs)
            corrs.append(sub_corrs)
            stdvs.append(sub_stdv)
        return corrs, stdvs

    def calculate_euclidean_oriented_dimer_correlation_function(self):
        """Get correlations between specific orientations of dimers."""
        nesw_mask = self.get_nesw_dimer_mask()
        nwse_mask = self.get_nwse_dimer_mask()
        correlations = self.get_dimer_correlations()
        masks = [(nesw_mask, nesw_mask), (nesw_mask, nwse_mask), (nwse_mask, nesw_mask), (nwse_mask, nwse_mask)]
        corrs = []
        stdvs = []
        for mask_a, mask_b in masks:
            sub_corrs = correlations[numpy.ix_(mask_a, mask_b)]
            sub_distances = self.graph.dimer_euclidean_distances[numpy.ix_(mask_a, mask_b)]
            sub_corrs, sub_stdv = average_by_distance(sub_distances, sub_corrs)
            corrs.append(sub_corrs)
            stdvs.append(sub_stdv)
        return corrs, stdvs

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

                orientation = get_variable_orientation(vara, varb)

                # TODO fix this whenever periodic is really working.
                front = graphbuilder.is_front(vara, vars_per_cell=self.graph.vars_per_cell)

                dax, day = graphbuilder.calculate_variable_direction(vara, vars_per_cell=self.graph.vars_per_cell,
                                                                     unit_cells_per_row=self.graph.unit_cells_per_row)
                dbx, dby = graphbuilder.calculate_variable_direction(varb, vars_per_cell=self.graph.vars_per_cell,
                                                                     unit_cells_per_row=self.graph.unit_cells_per_row)
                for dx, dy in [(dax, day), (dbx, dby)]:
                    ox, oy = cx + dx, cy + dy
                    connection = tuple(sorted(((cx, cy, front), (ox, oy, front))))
                    if connection in connection_indices:
                        j = connection_indices[connection]
                        self.dimer_to_flippable_matrix[i, j] = orientation
        return self.dimer_to_flippable_matrix

    def get_flippable_squares(self):
        dimers = self.get_dimer_matrix()
        dimer_to_flippable = self.get_dimer_to_flippable_squares_matrix()
        # Check that both adjacent are in the same direction.
        ori_dimers_adjacent_to_flippables = numpy.matmul(dimer_to_flippable.T, dimers == 1)
        num_dimers_adjacent_to_flippables = numpy.matmul(numpy.abs(dimer_to_flippable.T), dimers == 1)
        flippable_states = numpy.logical_and(num_dimers_adjacent_to_flippables == 2,
                                             numpy.abs(ori_dimers_adjacent_to_flippables) == 2)
        return flippable_states

    def get_defects(self):
        """Return a matrix of defects"""
        return self.get_dimer_vertex_counts() > 1

    def get_defect_correlations(self):
        """Get dimer vertex correlations in terms of defects."""
        if self.defect_correlations is None:
            # 1 for defects -1 for non-defects
            defects = self.get_defects() * 2 - 1
            self.defect_correlations = calculate_correlation_matrix(defects)
        return self.defect_correlations

    def calculate_euclidean_defect_correlation_function(self):
        """
        Calculate correlations as a function of euclidean distance for each dimer vertex.
        :return: NxN matrix of defect correlations, NxD matrix of defect distance correlations, and NxD matrix of
        standard deviations on the distance correlations, array of N ints which maps index to dimer vertex index.
        """
        defect_corr = self.get_defect_correlations()
        if self.defect_euclidean_distance_correlations is None or self.defect_euclidean_distance_stdv is None:
            distance_corrs, distance_stdv = average_by_distance(self.graph.dimer_vertex_distances, defect_corr)
            self.defect_euclidean_distance_correlations = distance_corrs
            self.defect_euclidean_distance_stdv = distance_stdv
        return defect_corr, self.defect_euclidean_distance_correlations, self.defect_euclidean_distance_stdv, self.graph.dimer_vertex_list

    def get_heightmaps(self):

        # To assign a height to each vertex, make the path through the vertices, passing through each edge.
        # treat the vertices connecting unit cells as a single vertex, since in the perfect dimer ground states they
        # should not have broken bonds.

        # All the not diagonal edges are effectively the vertices which can have height values
        effective_height_locations = [edge for edge, is_diagonal in zip(self.graph.sorted_edges,
                                                                        self.get_diagonal_dimer_mask())
                                      if not is_diagonal]


        # Make a matrix from diagonal edges to vertices

        diagonal_edges = [edge for edge, is_diagonal in zip(self.graph.sorted_edges, self.get_diagonal_dimer_mask())
                          if is_diagonal]
        diagonals = self.get_diagonal_dimer_matrix()




        # TODO fill this out.
        pass


def get_variable_orientation(var_a, var_b):
    """Returns 0 for vertical, +-1 for diagonal, None for unexpected."""
    dx_a, dy_a = graphbuilder.calculate_variable_direction(var_a)
    dx_b, dy_b = graphbuilder.calculate_variable_direction(var_b)
    # Vertical and horizontal bonds are green (and rare)
    if dx_a == -dx_b or dy_a == -dy_b:
        return 0
    if dx_a == dy_b and dy_a == dx_b:
        return 1
    if dx_a == -dy_b and dy_a == -dx_b:
        return -1
    return None


def average_by_distance(distance_matrix, values_matrix, binsize=1):
    """

    :param distance_matrix: NxN matrix of scalar distances.
    :param values_matrix: NxN matrix of scalar values.
    :param binsize: size of bins for output
    :return:
    """
    valid_distances = numpy.logical_not(numpy.logical_or(numpy.isinf(distance_matrix), numpy.isnan(distance_matrix)))
    max_distance = numpy.max(distance_matrix[valid_distances])
    num_bins = int(numpy.ceil(max_distance / binsize)) + 1
    num_vars = distance_matrix.shape[0]
    distance_values = numpy.zeros((num_vars, num_bins))
    distance_stdv = numpy.zeros((num_vars, num_bins))
    for i in range(num_vars):
        # Sum and count
        totals = numpy.zeros(num_bins)
        for j in range(num_vars):
            if valid_distances[i, j]:
                value = values_matrix[i, j]
                d = distance_matrix[i, j]
                d_index = int(numpy.floor(d / binsize))
                totals[d_index] += 1
                distance_values[i, d_index] += value
        # Average the values
        distance_values[i, :] = distance_values[i, :] / totals
        # Calculate variance
        for j in range(num_vars):
            if valid_distances[i, j]:
                value = values_matrix[i, j]
                d = distance_matrix[i, j]
                d_index = int(numpy.floor(d / binsize))
                distance_stdv[i, d_index] += (value - distance_values[i, d_index]) ** 2
        # Calculate standard deviation from variance
        distance_stdv[i, :] = numpy.sqrt(distance_stdv[i, :] / totals)
    return distance_values, distance_stdv


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


def calculate_correlation_matrix(variable_values, other_variables=None):
    """
    Calculates the cosine similarity between each set of variables over time.
    :param variable_values: nxd matrix of n variables and their values across d runs.
    :param other_variables: mxd matrix of m variables and their values across d runs (defaults to variable_values).
    :return: nxm matrix of the cosine similarities
    """
    variable_values = variable_values * 1.0
    # Subtract mean
    mean_per_var = numpy.mean(variable_values, axis=-1)
    variable_values = variable_values - numpy.expand_dims(mean_per_var, axis=-1)

    if other_variables is None:
        other_variables = variable_values
    else:
        mean_per_var = numpy.mean(other_variables, axis=-1)
        other_variables = other_variables - numpy.expand_dims(mean_per_var, axis=-1)

    # Correlations
    m = numpy.matmul(variable_values, other_variables.T)

    v_sqr = numpy.square(variable_values)
    v_norm_sqr = numpy.sum(v_sqr, -1)
    v_norms = numpy.sqrt(v_norm_sqr)

    v_sqr = numpy.square(other_variables)
    v_norm_sqr = numpy.sum(v_sqr, -1)
    other_norms = numpy.sqrt(v_norm_sqr)

    norm_mat = numpy.outer(v_norms, other_norms)
    norm_m = m / norm_mat
    return norm_m


def edge_is_satisfied(variable_values, edges, vara, varb):
    """Returns true if edge is in minimum energy state."""
    j_val = edges[(min(vara, varb), max(vara, varb))]
    if j_val > 0:
        return variable_values[vara] != variable_values[varb]
    else:
        return variable_values[vara] == variable_values[varb]
