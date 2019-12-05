import numpy
import graphbuilder


class GraphAnalyzer:
    def __init__(self, edges, samples, vars_per_cell=8, unit_cells_per_row=16):
        # I don't want to handle rear yet
        if not all(is_front(v) for edge in edges for v in edge):
            raise NotImplemented("Does not yet handle rear.")

        self.edges = edges
        self.samples = samples
        self.vars_per_cell = vars_per_cell
        self.unit_cells_per_row = unit_cells_per_row
        # Memoization for later
        self.var_map = None
        self.var_mat = None
        self.vertex_distances = None
        self.distance_lookup = None
        self.variable_correlations = None
        self.variable_distance_correlations = None
        self.variable_distance_stdv = None
        self.dimer_matrix = None
        self.edge_lookup = None
        self.dimer_correlation = None
        self.defect_matrix = None
        self.unit_cells = None
        self.unit_cell_bounding_box = None
        self.all_vars = None
        self.dimer_vertex_list = None
        self.edge_to_vertex_matrix = None
        self.vertex_counts = None

    def get_all_vars(self):
        """Return sorted list of all variables in the graph"""
        if self.all_vars is None:
            self.all_vars = list(sorted(set(v for edge in self.edges for v in edge)))
        return self.all_vars

    def get_unit_cells(self):
        """Return sorted list of all unit cells in graph. (x, y, is_front)"""
        if self.unit_cells is None:
            all_vars = self.get_all_vars()
            var_traits = (get_var_traits(var_indx, self.vars_per_cell, self.unit_cells_per_row)
                          for var_indx in all_vars)
            self.unit_cells = list(sorted(set((cx, cy, is_front(indx, self.vars_per_cell))
                                              for cx, cy, indx in var_traits)))

            minx, miny = self.unit_cells_per_row, self.unit_cells_per_row
            maxx, maxy = 0, 0
            for cx, cy, _ in self.unit_cells:
                minx = min(minx, cx)
                miny = min(miny, cy)
                maxx = max(maxx, cx)
                maxy = max(maxy, cy)
                self.unit_cell_bounding_box = ((minx, miny), (maxx, maxy))
        return self.unit_cells

    def get_flat_samples(self):
        if self.var_map is None or self.var_mat is None:
            var_map, var_mat = flatten_dicts(self.samples)
            self.var_map = var_map
            self.var_mat = var_mat
        return self.var_map, self.var_mat

    def get_vertex_distances(self):
        if self.vertex_distances is None or self.distance_lookup:
            distances, all_vars = variable_distances(self.edges)
            self.vertex_distances = distances
            self.distance_lookup = all_vars
        return self.vertex_distances, self.distance_lookup

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
        distances, all_vars = self.get_vertex_distances()
        var_corr = self.get_correlation_matrix()
        if self.variable_distance_correlations is None or self.variable_distance_stdv is None:
            # max distance (protected from infinity)
            max_dist = min(numpy.max(distances), len(self.edges))
            distance_corrs = numpy.zeros((len(all_vars), max_dist + 1))
            distance_stdv = numpy.zeros((len(all_vars), max_dist + 1))
            # Need to find a numpy-ish way to do this stuff.
            for i, v in enumerate(all_vars):
                totals = numpy.zeros(max_dist + 1)
                for j in range(len(all_vars)):
                    corr = var_corr[i, j]
                    d = distances[i, j]
                    if d <= max_dist:
                        totals[d] += 1
                        distance_corrs[i, d] += corr
                distance_corrs[i, :] = distance_corrs[i, :] / totals
                for j in range(len(all_vars)):
                    corr = var_corr[i, j]
                    d = distances[i, j]
                    if d <= max_dist:
                        distance_stdv[i, d] += (corr - distance_corrs[i, d]) ** 2
                distance_stdv[i, :] = numpy.sqrt(distance_stdv[i, :] / totals)
            self.variable_distance_correlations = distance_corrs
            self.variable_distance_stdv = distance_stdv
        return var_corr, self.variable_distance_correlations, self.variable_distance_stdv, all_vars

    def get_dimer_matrix(self):
        """Get matrix of dimers for each sample."""
        if self.dimer_matrix is None or self.edge_lookup is None:
            # Get flat versions of variables, values, and make lookup tables.
            var_map, var_mat = self.get_flat_samples()
            var_lookup = {v: k for k, v in enumerate(var_map)}
            sorted_edges = list(sorted(self.edges))
            edge_list = numpy.asarray([(var_lookup[a], var_lookup[b]) for a, b in sorted_edges])
            edge_values = numpy.asarray([self.edges[edge] for edge in sorted_edges])
            a_values = var_mat[edge_list[:, 0], :]
            b_values = var_mat[edge_list[:, 1], :]

            # Gives +1 if variables are equal, -1 if not
            spin_product = a_values * b_values
            # Gives +1 if edge is positive, -1 if not.
            edge_signs = numpy.expand_dims(numpy.sign(edge_values), -1)

            # If edge is positive, and product is negative (unequal), then satisfied.
            # So satisfied edges are negative, broken are positive
            self.dimer_matrix = spin_product * edge_signs
            self.edge_lookup = {edge: i for i, edge in enumerate(sorted_edges)}
        return self.edge_lookup, self.dimer_matrix

    def get_dimer_correlations(self):
        """
        Get the correlation of broken bonds (dimers).
        :param edges: map {(int, int): float} (edges and their couplings)
        :param samples: list of maps {int: float} for each sample
        :return: map from edge to index, and DxD matrix of correlations.
        """
        # Get dimers per sample
        edge_lookup, edges_broken = self.get_dimer_matrix()
        if self.dimer_correlation is None:
            # Now get correlations
            self.dimer_correlation = calculate_correlation_matrix(edges_broken)
        return edge_lookup, self.dimer_correlation

    def get_dimer_vertex_list(self):
        if self.dimer_vertex_list is None:
            # TODO update this to deal with front/rear correctly.
            # First all unit cells
            unit_cells = set(self.get_unit_cells())
            # Then all the squares of unit cells
            unit_cycles = set()
            for (cx, cy, front) in unit_cells:
                # Add the bottom-right one for each cell
                sides = [
                    (cx, cy, front),
                    (cx + 1, cy, front),
                    (cx + 1, cy + 1, front),
                    (cx, cy + 1, front)
                ]
                if all(side in unit_cells for side in sides[1:]):
                    unit_cycles.add(tuple(sorted(tuple(sides))))
            self.dimer_vertex_list = []
            self.dimer_vertex_list.extend(sorted(unit_cells))
            self.dimer_vertex_list.extend(sorted(unit_cycles))
        return self.dimer_vertex_list

    def get_edge_to_vertex_matrix(self):
        if self.edge_to_vertex_matrix is None:
            edge_lookup, _ = self.get_dimer_matrix()
            dimer_vertex_list = self.get_dimer_vertex_list()
            dimer_vertex_lookup = {k: i for i, k in enumerate(dimer_vertex_list)}
            self.edge_to_vertex_matrix = numpy.zeros((len(self.edges), len(dimer_vertex_list)), dtype=numpy.int8)
            for (va, vb) in sorted(self.edges):
                v1, v2 = get_dimer_vertices_for_edge(va, vb, vars_per_cell=self.vars_per_cell,
                                                     unit_cells_per_row=self.unit_cells_per_row)
                i = edge_lookup[(va, vb)]
                if v1 in dimer_vertex_lookup:
                    self.edge_to_vertex_matrix[i, dimer_vertex_lookup[v1]] = 1
                if v2 in dimer_vertex_lookup:
                    self.edge_to_vertex_matrix[i, dimer_vertex_lookup[v2]] = 1
        return self.edge_to_vertex_matrix

    def get_dimer_vertex_counts(self):
        if self.vertex_counts is None:
            # Get dimers per sample
            edge_lookup, edges_broken = self.get_dimer_matrix()
            vertex_list = self.get_dimer_vertex_list()
            edge_to_vertex_matrix = self.get_edge_to_vertex_matrix()
            self.vertex_counts = numpy.matmul(edge_to_vertex_matrix.T, edges_broken == 1)
        return self.vertex_counts


def get_dimer_vertices_for_edge(vara, varb, vars_per_cell=8, unit_cells_per_row=16):
    """Returns the two dimer vertices attached by this dimer."""
    acx, acy, rela = get_var_traits(vara, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    bcx, bcy, relb = get_var_traits(varb, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    front = is_front(vara)
    if not front:
        raise NotImplemented("Not implemented on rear.")
    if acx == bcx and acy == bcy:
        cx, cy = acx, acy
        adx, ady = calculate_variable_direction(vara, vars_per_cell=vars_per_cell,
                                                unit_cells_per_row=unit_cells_per_row)
        bdx, bdy = calculate_variable_direction(varb, vars_per_cell=vars_per_cell,
                                                unit_cells_per_row=unit_cells_per_row)
        unit_cycle = tuple(sorted([
            (cx, cy, front),
            (cx+adx, cy+ady, front),
            (cx+bdx, cy+bdy, front),
            (cx+adx+bdx, cy+ady+bdy, front)
        ]))
        return (cx, cy, front), unit_cycle
    else:
        # One should be zero, one should be +-1
        dx = bcx - acx
        dy = bcy - acy
        # Since the difference is along one axis, add and subtract the flipped to move along other axis.
        unit_cycle_a = tuple(sorted([
            (acx, acy, front),
            (bcx, bcy, front),
            (acx+dy, acy+dx, front),
            (bcx+dy, bcy+dx, front),
        ]))
        unit_cycle_b = tuple(sorted([
            (acx, acy, front),
            (bcx, bcy, front),
            (acx-dy, acy-dx, front),
            (bcx-dy, bcy-dx, front),
        ]))
        return unit_cycle_a, unit_cycle_b


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


def get_var_traits(index, vars_per_cell=8, unit_cells_per_row=16):
    """
    Get the relative unit cell indices and variable index from the absolute one
    :param index: absolute index
    :return: (unit_x, unit_y, relative_var)
    """
    var_relative = index % vars_per_cell
    unit_cell_index = index // vars_per_cell
    unit_cell_x = unit_cell_index % unit_cells_per_row
    unit_cell_y = unit_cell_index // unit_cells_per_row
    return unit_cell_x, unit_cell_y, var_relative


def is_front(index, vars_per_cell=8):
    """Returns if the absolute index is a front or rear unit cell."""
    var_relative = index % vars_per_cell
    return var_relative in [0, 1, 4, 5]


def variable_distances(edges):
    """
    Floyd's algorithm for all pairs shortest paths.
    :param edges: list of edges (var_a, var_b)
    :return: matrix of distances [var_a, var_b] and array of index to variable number
    """
    all_vars = numpy.asarray(list(sorted(set(v for vars in edges for v in vars))))
    var_lookup = {k: i for i, k in enumerate(all_vars)}
    n_vars = len(all_vars)
    # Maximum size to which we can safely multiply by 2.
    dist_mat = (numpy.zeros((n_vars, n_vars), dtype=numpy.uint32) - 1) // 2
    numpy.fill_diagonal(dist_mat, 0)
    for va, vb in edges:
        ia, ib = var_lookup[va], var_lookup[vb]
        dist_mat[ia, ib] = 1
        dist_mat[ib, ia] = 1
    for k in range(n_vars):
        for i in range(n_vars):
            for j in range(n_vars):
                if dist_mat[i, k] + dist_mat[k, j] < dist_mat[i, j]:
                    dist_mat[i, j] = dist_mat[i, k] + dist_mat[k, j]
    return dist_mat, all_vars


def is_unit_cell_type_a(unit_x, unit_y):
    """Returns if unit cell is of type A (true) or B (false)"""
    return ((unit_x + unit_y) % 2) == 0


def calculate_variable_direction(var_index, vars_per_cell=8, unit_cells_per_row=16):
    """Calculate the direction associated with the variable."""
    unit_x, unit_y, rel_var = get_var_traits(var_index, vars_per_cell=vars_per_cell,
                                             unit_cells_per_row=unit_cells_per_row)
    dx, dy, side = graphbuilder.Graph.var_connections[rel_var]
    # dx and dy are defined as A to B, so if B then reverse
    if not is_unit_cell_type_a(unit_x, unit_y):
        dx, dy = -dx, -dy
    return dx, dy


def edge_is_satisfied(variable_values, edges, vara, varb):
    """Returns true if edge is in minimum energy state."""
    j_val = edges[(min(vara, varb), max(vara, varb))]
    if j_val > 0:
        return variable_values[vara] != variable_values[varb]
    else:
        return variable_values[vara] == variable_values[varb]