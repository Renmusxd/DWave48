from matplotlib import pyplot
import numpy
from bathroom_tile import graphbuilder


def draw_graph(graph, save_to=None):
    dx = 0.2
    dy = 0.2

    offsets = {
        i: (-dx, y) for i, y in enumerate(numpy.linspace(-dy, dy, 4))
    }
    offsets.update({
        4+i: (dx, y) for i, y in enumerate(numpy.linspace(-dy, dy, 4))
    })

    min_ux = None
    max_ux = None
    min_uy = None
    max_uy = None
    var_xs = {}
    var_ys = {}
    for var in graph.all_vars:
        ux, uy, rel = graphbuilder.get_var_traits(var)
        rel_x, rel_y = offsets[rel]
        var_xs[var] = ux + rel_x
        var_ys[var] = uy + rel_y
        if min_ux is None or ux < min_ux:
            min_ux = ux
        if min_uy is None or uy < min_uy:
            min_uy = uy
        if max_ux is None or ux > max_ux:
            max_ux = ux
        if max_uy is None or uy > max_uy:
            max_uy = uy

    scatter_x = []
    scatter_y = []
    for x in range(min_ux, max_ux+1):
        for y in range(min_uy, max_uy+1):
            for rel in range(8):
                rel_x, rel_y = offsets[rel]
                scatter_x.append(x + rel_x)
                scatter_y.append(y + rel_y)
    pyplot.scatter(scatter_x, scatter_y, c='black')

    for (a, b), j in graph.edges.items():
        if j < 0.0:
            c = 'b'
        else:
            c = 'r'
        if abs(j) > 1.0:
            c = 'g'
        pyplot.plot([var_xs[a], var_xs[b]], [var_ys[a], var_ys[b]], c=c)
    pyplot.gca().invert_yaxis()
    if save_to:
        pyplot.savefig(save_to)
        pyplot.close()
    else:
        pyplot.show()


if __name__ == "__main__":
    gb = graphbuilder.GraphBuilder()
    start_x = 2
    start_y = 2
    nx = 4
    ny = 4
    for x in range(start_x, start_x+nx):
        for y in range(start_y, start_y+ny):
            gb.add_unit_cell(x, y, True)
            gb.add_unit_cell(x, y, False)
    gb.enable_dwave_periodic_boundary(((start_x, start_x+nx), (start_y, start_y+ny)), 5.0)
    gb.connect_all()
    graph = gb.build(calculate_distances=False, calculate_traits=False)
    draw_graph(graph)
