import numpy
from bathroom_tile import graphbuilder


class GraphAnalyzer:
    """Things about the graph using data from the samples."""

    def __init__(self, graph, var_map, var_mat, energies):
        self.graph = graph
        self.var_map = var_map
        self.var_mat = var_mat
        self.energies = energies

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

        self.nesw_dimer_matrix = None
        self.nwse_dimer_matrix = None

        # from get_defect_correlations
        self.defect_correlations = None

        self.defect_euclidean_distance_correlations = None
        self.defect_euclidean_distance_stdv = None

        # diagonals
        self.diagonal_dimer_euclidean_distance_correlations = None
        self.diagonal_dimer_euclidean_distance_stdv = None

        # For ease of use later.
        lowest_e = numpy.argmin(self.energies)
        sample = var_mat[:, lowest_e]
        self.lowest_e_sample = {k: sample[i] for i, k in enumerate(self.graph.all_vars)}

    def get_correlation_matrix(self):
        if self.variable_correlations is None:
            self.variable_correlations = calculate_correlation_matrix(self.var_mat)
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
            var_lookup = {v: k for k, v in enumerate(self.var_map)}
            edge_list = numpy.asarray([(var_lookup[a], var_lookup[b]) for a, b in self.graph.sorted_edges])
            edge_values = numpy.asarray([self.graph.edges[edge] for edge in self.graph.sorted_edges])
            a_values = self.var_mat[edge_list[:, 0], :]
            b_values = self.var_mat[edge_list[:, 1], :]

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
            unit_cells, _ = self.graph.calculate_unit_cells()
            unit_cells = set(unit_cells)
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

    def calculate_difference_order_parameter(self):
        mask = self.get_diagonal_dimer_mask()
        var_poss = numpy.asarray([graphbuilder.get_var_cartesian(a) for a in self.var_map])

        var_lookup = {v: k for k,v in enumerate(self.var_map)}
        diagonal_edges = numpy.asarray([(var_lookup[a], var_lookup[b])
                                        for (a, b), m in zip(self.graph.sorted_edges, mask) if m])
        diagonal_as, diagonal_bs = diagonal_edges[:, 0], diagonal_edges[:, 1]
        varxs, varys = var_poss[:, 0], var_poss[:, 1]
        edge_diff_xs = numpy.sign(varxs[diagonal_as] - varxs[diagonal_bs])
        edge_diff_ys = numpy.sign(varys[diagonal_as] - varys[diagonal_bs])
        tl_br_mask = edge_diff_xs == edge_diff_ys
        tr_bl_mask = edge_diff_xs != edge_diff_ys

        diagonal_dimers = self.get_diagonal_dimer_matrix()
        diagonal_dimers_tf = (diagonal_dimers + 1) / 2
        total_diagonal_dimers = numpy.sum(diagonal_dimers_tf, axis=0)

        tl_br = numpy.sum(numpy.expand_dims(tl_br_mask, -1) * diagonal_dimers_tf, axis=0)
        tr_bl = numpy.sum(numpy.expand_dims(tr_bl_mask, -1) * diagonal_dimers_tf, axis=0)

        return (tl_br - tr_bl) / (1.0*total_diagonal_dimers)

    def calculate_fourier_order_parameter(self, k_nesw=None, k_nwse=None):
        if k_nesw is None:
            k_nesw = numpy.asarray([1.0, 1.0])
        if k_nwse is None:
            k_nwse = numpy.asarray([1.0, -1.0])
        if type(k_nesw) == float:
            k_nesw = numpy.asarray([k_nesw, k_nesw])
        if type(k_nesw) == float:
            k_nwse = numpy.asarray([k_nwse, -k_nwse])

        mask = self.get_diagonal_dimer_mask()
        var_poss = numpy.asarray([graphbuilder.get_var_cartesian(a) for a in self.var_map])

        var_lookup = {v: k for k, v in enumerate(self.var_map)}
        diagonal_edges = numpy.asarray([(var_lookup[a], var_lookup[b])
                                        for (a, b), m in zip(self.graph.sorted_edges, mask) if m])
        diagonal_as, diagonal_bs = diagonal_edges[:, 0], diagonal_edges[:, 1]

        varxs, varys = var_poss[:, 0], var_poss[:, 1]

        edge_diff_xs = numpy.sign(varxs[diagonal_as] - varxs[diagonal_bs])
        edge_diff_ys = numpy.sign(varys[diagonal_as] - varys[diagonal_bs])
        # Dimers are perpendicular to the edge.
        nwse_mask = edge_diff_xs != edge_diff_ys
        nesw_mask = edge_diff_xs == edge_diff_ys

        diagonal_dimers = self.get_diagonal_dimer_matrix()
        diagonal_dimers_tf = (diagonal_dimers + 1) / 2

        nwse_tf = numpy.expand_dims(nwse_mask, -1) * diagonal_dimers_tf
        nesw_tf = numpy.expand_dims(nesw_mask, -1) * diagonal_dimers_tf

        edge_avg_xs = (varxs[diagonal_as] + varxs[diagonal_bs]) / 2.0
        edge_avg_ys = (varys[diagonal_as] + varys[diagonal_bs]) / 2.0

        # Broadcast to arbitrary shape.
        while len(k_nwse.shape) > len(edge_avg_xs.shape):
            edge_avg_xs = numpy.expand_dims(edge_avg_xs, axis=0)
            edge_avg_ys = numpy.expand_dims(edge_avg_ys, axis=0)

        nwse_phase = k_nwse[..., [0]] * edge_avg_xs + k_nwse[..., [1]] * edge_avg_ys
        nesw_phase = k_nesw[..., [0]] * edge_avg_xs + k_nesw[..., [1]] * edge_avg_ys

        nwse = numpy.exp(1.0j*nwse_phase) @ nwse_tf
        nesw = numpy.exp(1.0j*nesw_phase) @ nesw_tf

        total_diagonal_dimers = numpy.sum(diagonal_dimers_tf, axis=0)
        orders = numpy.stack([nwse/total_diagonal_dimers, nesw/total_diagonal_dimers])
        return orders

    def calculate_complex_angle_order_parameter(self):
        # Look just at unit cells and see which dimers break
        unit_cells, _ = self.graph.calculate_unit_cells()
        unit_cell_lookup = {(cx, cy): i for i, (cx, cy, _) in enumerate(unit_cells)}
        n_unit_cells = len(unit_cells)
        n_edges = len(self.graph.sorted_edges)
        edge_to_unit_cell_orientations = numpy.zeros((n_unit_cells, n_edges), dtype=numpy.complex128)
        edge_to_unit_cell_count = numpy.zeros((n_unit_cells, n_edges))

        for edge_indx, (var_a, var_b) in enumerate(self.graph.sorted_edges):
            unit_ax, unit_ay, _ = graphbuilder.get_var_traits(var_a)
            unit_bx, unit_by, _ = graphbuilder.get_var_traits(var_b)
            if unit_ax == unit_bx and unit_ay == unit_by:
                unit_cell_indx = unit_cell_lookup[(unit_ax, unit_ay)]
                edge_to_unit_cell_count[unit_cell_indx, edge_indx] = 1

                dx_a, dy_a = graphbuilder.calculate_variable_direction(var_a)
                dx_b, dy_b = graphbuilder.calculate_variable_direction(var_b)

                dx = dx_a + dx_b
                dy = dy_a + dy_b
                if (dx, dy) == (1, 1):
                    orientation = 1.0
                elif (dx, dy) == (1, -1):
                    orientation = 1.0j
                elif (dx, dy) == (-1, -1):
                    orientation = -1.0
                elif (dx, dy) == (-1, 1):
                    orientation = -1.0j
                else:
                    orientation = 0.0

                edge_to_unit_cell_orientations[unit_cell_indx, edge_indx] = orientation

        dimers = self.get_dimer_matrix()

        dimers = (dimers+1)/2

        dimer_counts = edge_to_unit_cell_count @ dimers
        unit_cell_orientations = edge_to_unit_cell_orientations @ dimers
        dimer_mask = dimer_counts == 1
        unit_cell_orientations = unit_cell_orientations * dimer_mask

        # We are just looking at the (pi,pi) momentum
        unit_cell_phases = numpy.exp(1.0j*numpy.pi*numpy.asarray([cx+cy for (cx, cy, _) in unit_cells]))
        pi_pi_fourier = unit_cell_orientations * numpy.expand_dims(unit_cell_phases, -1)
        return numpy.mean(pi_pi_fourier, axis=0)


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
        diagonal_lookup = {
            edge: i for i, edge in enumerate(diagonal_edges)
        }

        # Lets make a lookup function
        def unit_cell_for_var(v):
            x, y, _ = graphbuilder.get_var_traits(v, vars_per_cell=self.graph.vars_per_cell,
                                                  unit_cells_per_row=self.graph.unit_cells_per_row)
            return x, y
        lookup = {
            (unit_cell_for_var(va), unit_cell_for_var(vb)): i
            for i, (va, vb) in enumerate(effective_height_locations)
        }

        cumulative_matrix = numpy.zeros((len(effective_height_locations), len(diagonal_edges)))

        def px_direction(cell_a, cell_b):
            ax, ay = cell_a
            bx, by = cell_b
            if ax == bx:
                return (ax, ay), (bx + 1, by - 1)
            else:
                return (ax + 1, ay - 1), (bx, by)

        def mx_direction(cell_a, cell_b):
            ax, ay = cell_a
            bx, by = cell_b
            if ax == bx:
                return (ax - 1, ay + 1), (bx, by)
            else:
                return (ax, ay), (bx - 1, by + 1)

        # First lets fill out from the diagonal

        dx = 1
        dy = 0
        x, y = min(unit_cell_for_var(v) for vs in effective_height_locations for v in vs)
        key = ((x, y), (x + dx, y + dy))
        last_index = lookup[key]
        while key in lookup:
            # Get the "center" of the diagonal
            index = lookup[key]
            base_mult = 1 if graphbuilder.is_type_a(x, y) else -1
            if index != last_index:
                cumulative_matrix[index, :] = cumulative_matrix[last_index, :]
                # Get the dimer required to be crossed to get to x, y from x-dy, y-dx
                edge = get_connecting_diagonal_dimer(x, y, 2*dx - 1, 2*dy - 1)
                cumulative_matrix[index, diagonal_lookup[edge]] = base_mult
            # Now move in either direction
            for direction, dir_mult in zip([px_direction, mx_direction], [1, 1]):
                last_subkey = key
                sub_key = direction(key[0], key[1])
                mult = base_mult*dir_mult
                while sub_key in lookup:
                    cumulative_matrix[lookup[sub_key], :] = cumulative_matrix[lookup[last_subkey], :]
                    center_point, edge = get_diagonal_edge_for_unit_cell_edges(last_subkey, sub_key)
                    cumulative_matrix[lookup[sub_key], diagonal_lookup[edge]] = mult
                    mult = mult*-1

                    last_subkey = sub_key
                    sub_key = direction(*sub_key)

            last_index = index
            x, y = x+dx, y+dy
            dx, dy = dy, dx
            key = ((x, y), (x + dx, y + dy))

        diagonals = self.get_diagonal_dimer_matrix().copy()

        # Fill out with the default values for height calculations
        diagonals[diagonals == 1] = 3
        diagonals[diagonals == -1] = -1
        heights = numpy.matmul(cumulative_matrix, diagonals)
        return effective_height_locations, heights

    def get_charges(self, edge_to_vertex_matrix=None):
        dimer_matrix = self.get_dimer_matrix()
        diag_dimers = numpy.expand_dims(self.get_diagonal_dimer_mask(), axis=-1)

        e_fields = (dimer_matrix*2 + 1) * diag_dimers

        if edge_to_vertex_matrix is None:
            edge_to_vertex_matrix = self.graph.edge_to_vertex_matrix

        charges = edge_to_vertex_matrix.T @ e_fields
        charges = charges * numpy.expand_dims(numpy.asarray([(1 if len(v)==4 else -1) for v in self.graph.dimer_vertex_list]), axis=-1)

        return charges


def get_diagonal_edge_for_unit_cell_edges(edge_a, edge_b, front=True):
    dax, day = edge_a[1][0] - edge_a[0][0], edge_a[1][1] - edge_a[0][1]
    dbx, dby = edge_b[1][0] - edge_b[0][0], edge_b[1][1] - edge_b[0][1]
    if dax == 1 and day == 0 and dbx == 0 and dby == 1:
        edge_x = edge_a
        edge_y = edge_b
    elif dax == 0 and day == 1 and dbx == 1 and dby == 0:
        edge_x = edge_b
        edge_y = edge_a
    else:
        raise Exception("Could not get diagonal for unit cell edges {} and {}".format(edge_a, edge_b))
    for i, point_a in enumerate(edge_x):
        for j, point_b in enumerate(edge_y):
            if point_a == point_b:
                # not i ==> from -1 to 1 instead of 0, 1
                dx = 2*(1 - i) - 1
                dy = 2*(1 - j) - 1
                return point_a, get_connecting_diagonal_dimer(point_a[0], point_a[1], dx, dy, front=front)


def get_connecting_diagonal_dimer(x, y, dx, dy, front=True):
    if not abs(dx) == 1 or not abs(dy) == 1:
        raise Exception("This is not a proper diagonal, dx,dy=+-1")
    conn = graphbuilder.get_connection_cells()
    if not graphbuilder.is_type_a(x, y):
        dx, dy = -dx, -dy
    vara = conn[(dx, 0, front)]
    varb = conn[(0, dy, front)]

    vara = graphbuilder.var_num(x, y, vara)
    varb = graphbuilder.var_num(x, y, varb)

    return min(vara, varb), max(vara, varb)


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
    if num_bins > 1e6:
        raise Exception("Too many distance bins, this can't be good.")
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

        totals = numpy.maximum(totals, numpy.ones_like(totals))
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
