import io
import numpy
import graphanalysis
import graphbuilder


def relative_pos_of_relative_index(relative_index):
    if relative_index == 0 or relative_index == 2:
        return 0, -1
    elif relative_index == 1 or relative_index == 3:
        return 0, 1
    elif relative_index == 4 or relative_index == 6:
        return -1, 0
    elif relative_index == 5 or relative_index == 7:
        return 1, 0


def position_of_unit_cell(x, y, dist=10):
    return dist * x, dist * y


def get_var_pos(index, vars_per_cell=8, unit_cells_per_row=16, dist=10):
    unit_cell_x, unit_cell_y, var_relative = graphanalysis.get_var_traits(index, vars_per_cell, unit_cells_per_row)

    x, y = position_of_unit_cell(unit_cell_x, unit_cell_y, dist=dist)
    dx, dy = relative_pos_of_relative_index(var_relative)
    if not graphbuilder.is_type_a(unit_cell_x, unit_cell_y):
        dx = -dx
        dy = -dy
    return x + dx, y + dy


def make_edges_svg(edges, unit_cells_per_row=16, vars_per_cell=8, dist=5, color_fn=None, front=True):
    contents, _ = make_edges_contents(edges, unit_cells_per_row=unit_cells_per_row, vars_per_cell=vars_per_cell,
                                      dist=dist, color_fn=color_fn, front=front)
    if contents:
        return '<svg height="500.0pt" version="1.1" width="500.0pt" viewBox="0 0 1 1" ' \
               'xmlns="http://www.w3.org/2000/svg">\n{}</svg>'.format(
            contents
        )
    return None


def make_edges_contents(edges, unit_cells_per_row=16, vars_per_cell=8, dist=5, color_fn=None, front=True):
    def default_color_fn(_, __):
        return "rgb(0,0,0)"

    if color_fn is None:
        color_fn = default_color_fn

    minx = 1e32
    maxx = 0
    miny = 1e32
    maxy = 0

    debug_details_values = []
    to_draw_values = []
    for (var_a, var_b) in edges:
        if graphanalysis.is_front(var_a) != front or graphanalysis.is_front(var_b) != front:
            continue
        x1, y1 = get_var_pos(var_a, unit_cells_per_row=unit_cells_per_row, vars_per_cell=vars_per_cell, dist=dist)
        x2, y2 = get_var_pos(var_b, unit_cells_per_row=unit_cells_per_row, vars_per_cell=vars_per_cell, dist=dist)
        minx = min(x1, x2, minx)
        maxx = max(x1, x2, maxx)
        miny = min(y1, y2, miny)
        maxy = max(y1, y2, maxy)

        a_cell_x, a_cell_y, rel_var_a = graphanalysis.get_var_traits(var_a, unit_cells_per_row=unit_cells_per_row,
                                                                     vars_per_cell=vars_per_cell)
        b_cell_x, b_cell_y, rel_var_b = graphanalysis.get_var_traits(var_b, unit_cells_per_row=unit_cells_per_row,
                                                                     vars_per_cell=vars_per_cell)

        debug_details_values.append((a_cell_x, a_cell_y, b_cell_x, b_cell_y, rel_var_a, rel_var_b, var_a, var_b))
        to_draw_values.append((x1, y1, x2, y2, color_fn(var_a, var_b)))

    def normalize(x, y):
        lx = minx - 1
        ly = miny - 1
        mx = maxx + 1
        my = maxy + 1
        return (x - lx) / float(mx - lx), (y - ly) / float(my - ly)

    output = io.StringIO()
    for to_draw, to_detail in zip(to_draw_values, debug_details_values):
        a_cell_x, a_cell_y, b_cell_x, b_cell_y, rel_var_a, rel_var_b, var_a, var_b = to_detail
        x1, y1, x2, y2, color = to_draw

        x1, y1 = normalize(x1, y1)
        x2, y2 = normalize(x2, y2)

        title_text = "Going from unit cell ({},{}) to ({},{}) via {}->{} (raw vars: {}/{})".format(
            a_cell_x, a_cell_y, b_cell_x, b_cell_y, rel_var_a, rel_var_b, var_a, var_b
        )
        output.write("<!-- {} -->\n".format(title_text))
        output.write(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:{};stroke-width:0.01"><title>{}</title></line>\n'.format(
                x1, y1, x2, y2, color, title_text
            ))
    return output.getvalue(), normalize


def make_dimer_contents(broken_edges, normalize, unit_cells_per_row=16, vars_per_cell=8, dist=5,
                        dimer_color="red", width="0.01", front=True):
    dimer_positions = []
    for var_a, var_b in broken_edges:
        if graphanalysis.is_front(var_a) != front or graphanalysis.is_front(var_b) != front:
            continue
        a_cell_x, a_cell_y, rel_var_a = graphanalysis.get_var_traits(var_a, unit_cells_per_row=unit_cells_per_row,
                                                                     vars_per_cell=vars_per_cell)
        b_cell_x, b_cell_y, rel_var_b = graphanalysis.get_var_traits(var_b, unit_cells_per_row=unit_cells_per_row,
                                                                     vars_per_cell=vars_per_cell)
        var_a_pos = numpy.asarray(get_var_pos(var_a, unit_cells_per_row=unit_cells_per_row,
                                              vars_per_cell=vars_per_cell, dist=dist))
        var_b_pos = numpy.asarray(get_var_pos(var_b, unit_cells_per_row=unit_cells_per_row,
                                              vars_per_cell=vars_per_cell, dist=dist))
        middle = (var_a_pos + var_b_pos) / 2.0
        # If breaking across two cells, then vertical / horizontal dimer
        if rel_var_a == rel_var_b:
            # Dimer crosses the middle of the two, goes dist/2 on either side perpendicular to the direction
            diff_var = var_a_pos - var_b_pos
            diff_var = (diff_var / numpy.linalg.norm(diff_var)) * dist / 2.0
            diff_swap = numpy.asarray([diff_var[1], diff_var[0]])
            start_pos = middle - diff_swap
            end_pos = middle + diff_swap
            start_pos = normalize(start_pos[0], start_pos[1])
            end_pos = normalize(end_pos[0], end_pos[1])
            dimer_positions.append((start_pos, end_pos, (var_a, var_b)))

        # If internal to a cell, then diagonal
        elif (a_cell_x, a_cell_y) == (b_cell_x, b_cell_y):
            unit_cell_pos = numpy.asarray(position_of_unit_cell(a_cell_x, a_cell_y, dist=dist))

            # Dimer goes from unit_cell_pos, through the middle of var_a_pos, var_b_pos, then forward sqrt(2*dist)
            direction = middle - unit_cell_pos
            direction = direction / numpy.linalg.norm(direction) * numpy.sqrt(2.0 * dist)
            end_pos = direction + unit_cell_pos
            # Normalize to 0-1
            start = normalize(unit_cell_pos[0], unit_cell_pos[1])
            stop = normalize(end_pos[0], end_pos[1])
            dimer_positions.append((start, stop, (var_a, var_b)))
        else:
            raise Exception("Not sure how to draw dimer across {}-{}".format(var_a, var_b))

    output = io.StringIO()
    for (start_x, start_y), (stop_x, stop_y), (var_a, var_b) in dimer_positions:
        title_text = "Dimer across edge between sites {}-{}".format(var_a, var_b)
        output.write(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:{};stroke-width:{}"><title>{}</title></line>\n'.format(
                start_x, start_y, stop_x, stop_y, dimer_color, width, title_text
            ))
    return output.getvalue()


def make_dimer_svg(js, var_vals, unit_cells_per_row=16, vars_per_cell=8, dist=5,
                   edge_color="gray", dimer_edge_color="black",
                   dimer_color="red", front=True):
    def all_grey(_, __):
        return edge_color

    def is_not_broken(vara, varb):
        j_val = js[(min(vara, varb), max(vara, varb))]
        if j_val > 0:
            return var_vals[vara] != var_vals[varb]
        else:
            return var_vals[vara] == var_vals[varb]

    edges = list(js)
    broken_edges = [edge for edge in edges if not is_not_broken(edge[0], edge[1])]
    background_edges_contents, normalize = make_edges_contents(edges, unit_cells_per_row=unit_cells_per_row,
                                                               vars_per_cell=vars_per_cell, dist=dist,
                                                               color_fn=all_grey, front=front)
    if background_edges_contents:
        background_dimer_contents = make_dimer_contents(edges, normalize, unit_cells_per_row=unit_cells_per_row,
                                                        vars_per_cell=vars_per_cell, dist=dist,
                                                        dimer_color=dimer_edge_color, width="0.005", front=front)
        dimer_contents = make_dimer_contents(broken_edges, normalize, unit_cells_per_row=unit_cells_per_row,
                                             vars_per_cell=vars_per_cell, dist=dist, dimer_color=dimer_color, front=front)
        return '<svg height="500.0pt" version="1.1" width="500.0pt" viewBox="0 0 1 1" ' \
               'xmlns="http://www.w3.org/2000/svg">\n{}\n{}\n{}</svg>'.format(
            background_edges_contents,
            background_dimer_contents,
            dimer_contents)
    else:
        return None

def make_combined_dimer_svg(js, var_vals, unit_cells_per_row=16, vars_per_cell=8, dist=5,
                            edge_color="gray", dimer_edge_color="black",
                            front_dimer_color="red", rear_dimer_color="blue"):
    """Assuming lattice is the same front and back"""
    def all_grey(_, __):
        return edge_color

    def is_not_broken(vara, varb):
        j_val = js[(min(vara, varb), max(vara, varb))]
        if j_val > 0:
            return var_vals[vara] != var_vals[varb]
        else:
            return var_vals[vara] == var_vals[varb]

    edges = list(js)
    broken_edges = [edge for edge in edges if not is_not_broken(edge[0], edge[1])]
    background_edges_contents, normalize = make_edges_contents(edges, unit_cells_per_row=unit_cells_per_row,
                                                               vars_per_cell=vars_per_cell, dist=dist,
                                                               color_fn=all_grey, front=True)  # front and rear the same
    background_dimer_contents = make_dimer_contents(edges, normalize, unit_cells_per_row=unit_cells_per_row,
                                                    vars_per_cell=vars_per_cell, dist=dist,
                                                    dimer_color=dimer_edge_color, width="0.005", front=True)
    if background_edges_contents:
        rear_dimer_contents = make_dimer_contents(broken_edges, normalize, unit_cells_per_row=unit_cells_per_row,
                                                  vars_per_cell=vars_per_cell, dist=dist, dimer_color=rear_dimer_color,
                                                  width="0.015", front=False)
        front_dimer_contents = make_dimer_contents(broken_edges, normalize, unit_cells_per_row=unit_cells_per_row,
                                                   vars_per_cell=vars_per_cell, dist=dist,
                                                   dimer_color=front_dimer_color,
                                                   width="0.01", front=True)
        return '<svg height="500.0pt" version="1.1" width="500.0pt" viewBox="0 0 1 1" ' \
               'xmlns="http://www.w3.org/2000/svg">\n{}\n{}\n{}\n{}</svg>'.format(
            background_edges_contents,
            background_dimer_contents,
            rear_dimer_contents,
            front_dimer_contents)
    else:
        return None