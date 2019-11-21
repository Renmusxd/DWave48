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
    var_relative = index % vars_per_cell
    unit_cell_index = index // vars_per_cell
    unit_cell_x = unit_cell_index % unit_cells_per_row
    unit_cell_y = unit_cell_index // unit_cells_per_row
    return unit_cell_x, unit_cell_y, var_relative