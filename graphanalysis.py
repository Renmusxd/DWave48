import numpy
import collections


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
    variable_values = variable_values*1.0
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
    all_vars = list(sorted(set(v for vars in edges for v in vars)))
    var_lookup = {k: i for i,k in enumerate(all_vars)}
    n_vars = len(all_vars)
    # Maximum size to which we can safely multiply by 2.
    dist_mat = (numpy.zeros((n_vars, n_vars), dtype=numpy.uint32) - 1)//2
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


def calculate_correlation_function(edges, samples):
    var_map, var_mat = flatten_dicts(samples)
    distances, all_vars = variable_distances(edges)
    var_corr = calculate_correlation_matrix(var_mat)
    # max distance (protected from infinity)
    max_dist = min(numpy.max(distances), len(edges))
    distance_corrs = numpy.zeros((len(all_vars), max_dist+1))
    distance_stdv = numpy.zeros((len(all_vars), max_dist+1))
    for i,v in enumerate(all_vars):
        totals = numpy.zeros(max_dist+1)
        for j in range(len(all_vars)):
            corr = var_corr[i,j]
            d = distances[i,j]
            if d <= max_dist:
                totals[d] += 1
                distance_corrs[i, d] += corr
        distance_corrs[i,:] = distance_corrs[i,:] / totals
        for j in range(len(all_vars)):
            corr = var_corr[i,j]
            d = distances[i,j]
            if d <= max_dist:
                distance_stdv[i, d] += (corr - distance_corrs[i,d])**2
        distance_stdv[i,:] = numpy.sqrt(distance_stdv[i,:] / totals)
    return var_corr, distance_corrs, distance_stdv, all_vars