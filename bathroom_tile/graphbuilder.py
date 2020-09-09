import numpy
import scipy.spatial
import collections
import pickle
import os
from enum import Enum


def get_connection_cells():
    """Gives direction in terms of A to B"""
    return {
        # Front
        (0, -1, True): 0,
        (0, 1, True): 1,
        (-1, 0, True): 4,
        (1, 0, True): 5,
        # Rear
        (0, -1, False): 2,
        (0, 1, False): 3,
        (-1, 0, False): 6,
        (1, 0, False): 7,
    }


def get_var_to_connection_map():
    """Gives direction in terms of A to B"""
    return {
        v: k for k, v in get_connection_cells().items()
    }


class LatticeVar(Enum):
    A = 1
    B = 2
    C = 3
    D = 4

    @staticmethod
    def get_from_rel_var(cx, cy, rel_var):
        dx, dy, front = get_var_to_connection_map()[rel_var]

        if not is_type_a(cx, cy):
            dx = -dx
            dy = -dy

        if dx == 0 and dy == -1:
            return LatticeVar.A, front
        elif dx == 1 and dy == 0:
            return LatticeVar.B, front
        elif dx == 0 and dy == 1:
            return LatticeVar.C, front
        elif dx == -1 and dy == 0:
            return LatticeVar.D, front

    def get_rel_var(self, cx, cy, front=True):
        conns = get_connection_cells()
        if self == LatticeVar.A:
            dx = 0
            dy = -1
        elif self == LatticeVar.B:
            dx = 1
            dy = 0
        elif self == LatticeVar.C:
            dx = 0
            dy = 1
        elif self == LatticeVar.D:
            dx = -1
            dy = 0
        else:
            raise ValueError("Enum error: {}".format(self))

        if not is_type_a(cx, cy):
            dx = -dx
            dy = -dy

        return conns[(dx, dy, front)]


class GraphBuilder:
    connection_cells = get_connection_cells()
    var_connections = get_var_to_connection_map()

    def __init__(self, j=1.0, clone_strength=None):
        self.unit_cells = set()
        self.j = j
        self.clone = clone_strength
        self.existing_cell_connections = set()
        self.dwave_periodic_boundary = None

    def enable_dwave_periodic_boundary(self, enclose_rect, clone_j=1.0):
        self.dwave_periodic_boundary = enclose_rect
        self.clone = clone_j

    def build(self, h=0, ideal_periodic_boundaries=False, calculate_traits=True, calculate_distances=True, ideal_clones=False):
        connections = {}
        for x, y, front in self.unit_cells:
            if is_type_a(x, y):
                connections.update(make_unit_cell_a(x, y, j=self.j, front=front))
            else:
                connections.update(make_unit_cell_b(x, y, j=self.j, front=front))

        for cell_one, cell_two, front in self.existing_cell_connections:
            first_cell_type = ((cell_one[0] + cell_one[1]) % 2) == 0
            if first_cell_type:
                cella, cellb = cell_one, cell_two
            else:
                cella, cellb = cell_two, cell_one
            dx = cellb[0] - cella[0]
            dy = cellb[1] - cella[1]
            if (dx, dy, front) in GraphBuilder.connection_cells:
                v = GraphBuilder.connection_cells[(dx, dy, front)]
                abs_va = var_num(cella[0], cella[1], v)
                abs_vb = var_num(cellb[0], cellb[1], v)
                # Enter them in sorted order
                min_v = min(abs_va, abs_vb)
                max_v = max(abs_va, abs_vb)
                edge = (min_v, max_v)
                connections.update({edge: -self.j})

        # Default hs are same for all variables
        if h:
            hs = {v: h for v in set(v for vs in connections for v in vs)}
        else:
            hs = {}

        cloned_cells = None

        if ideal_periodic_boundaries:
            horizontal_ends = {}
            vertical_ends = {}

            for x, y, front in self.unit_cells:
                if (y, front) not in horizontal_ends:
                    horizontal_ends[(y, front)] = ((x, y), (x, y))
                else:
                    (min_x, min_y), (max_x, max_y) = horizontal_ends[(y, front)]
                    # Order by x
                    new_min_x, new_min_y = min((min_x, min_y), (x, y))
                    new_max_x, new_max_y = max((max_x, max_y), (x, y))
                    horizontal_ends[(y, front)] = ((new_min_x, new_min_y), (new_max_x, new_max_y))

                if (x, front) not in vertical_ends:
                    vertical_ends[(x, front)] = ((x, y), (x, y))
                else:
                    (min_x, min_y), (max_x, max_y) = vertical_ends[(x, front)]
                    # Order by y
                    new_min_y, new_min_x = min((min_y, min_x), (y, x))
                    new_max_y, new_max_x = max((max_y, max_x), (y, x))
                    vertical_ends[(x, front)] = ((new_min_x, new_min_y), (new_max_x, new_max_y))

            periodic_connections = {}
            def get_mult(x, y):
                if is_type_a(x, y):
                    return 1
                return -1

            for (_, front), ((ax, ay), (bx, by)) in horizontal_ends.items():
                # ax < bx
                va = GraphBuilder.connection_cells[(-1*get_mult(ax, ay), 0, front)]
                vb = GraphBuilder.connection_cells[(1*get_mult(bx, by), 0, front)]
                abs_va = var_num(ax, ay, va)
                abs_vb = var_num(bx, by, vb)

                min_v = min(abs_va, abs_vb)
                max_v = max(abs_va, abs_vb)
                edge = (min_v, max_v)
                periodic_connections.update({edge: -self.j})

            for (_, front), ((ax, ay), (bx, by)) in vertical_ends.items():
                # ay < by
                va = GraphBuilder.connection_cells[(0, -1*get_mult(ax, ay), front)]
                vb = GraphBuilder.connection_cells[(0, 1*get_mult(bx, by), front)]
                abs_va = var_num(ax, ay, va)
                abs_vb = var_num(bx, by, vb)

                min_v = min(abs_va, abs_vb)
                max_v = max(abs_va, abs_vb)
                edge = (min_v, max_v)
                periodic_connections.update({edge: -self.j})
            connections.update(periodic_connections)
        elif self.dwave_periodic_boundary is not None:
            (min_x, max_x), (min_y, max_y) = self.dwave_periodic_boundary
            cloned_cells = set()
            for ux in range(min_x, max_x):
                bond_j = self.j
                if ux % 2 != 0:
                    bond_j = -bond_j
                connections.update(make_periodic_connection((ux, min_y-1), (ux, min_y), bond_j, -self.clone,
                                                            ideal_clones=ideal_clones))
                connections.update(make_periodic_connection((ux, max_y), (ux, max_y-1), bond_j, -self.clone,
                                                            ideal_clones=ideal_clones))
                cloned_cells.add((ux, min_y - 1))
                cloned_cells.add((ux, max_y))

            for uy in range(min_y, max_y):
                connections.update(make_periodic_connection((min_x-1, uy), (min_x, uy), -self.j, -self.clone,
                                                            ideal_clones=ideal_clones))
                connections.update(make_periodic_connection((max_x, uy), (max_x-1, uy), -self.j, -self.clone,
                                                            ideal_clones=ideal_clones))
                cloned_cells.add((min_x-1, uy))
                cloned_cells.add((max_x, uy))

        return Graph(connections, hs, calculate_distances=calculate_distances,
                     calculate_traits=calculate_traits, periodic_boundaries=ideal_periodic_boundaries,
                     clone_j=self.clone, cloned_cells=cloned_cells)

    def add_cells(self, cells, fronts=None):
        if fronts is None:
            fronts = True
        if type(fronts) == bool:
            front = bool(fronts)
            fronts = (front for _ in range(len(cells)))
        for (x, y), front in zip(cells, fronts):
            self.add_unit_cell(x, y, front=front)

    def add_cell(self, *cells):
        self.add_cells(cells)

    def add_unit_cell(self, x, y, front=True):
        self.unit_cells.add((x, y, front))

    def connect_cells(self, cella, cellb, front=True):
        minc = min(cella, cellb)
        maxc = max(cella, cellb)
        self.existing_cell_connections.add((minc, maxc, front))

    def connect_all(self):
        # Now add connections
        for x, y, front in self.unit_cells:
            for dx, dy, conn_side in GraphBuilder.connection_cells:
                if front != conn_side:
                    continue
                ox = x + dx
                oy = y + dy
                if (ox, oy, front) in self.unit_cells:
                    self.connect_cells((x, y), (ox, oy), front=front)


class Graph:
    """Aspects of the graph which do not rely on data"""

    def __init__(self, edges, hs, graph_cache_dir='graphcache', vars_per_cell=8, unit_cells_per_row=16,
                 calculate_traits=True, periodic_boundaries=False, calculate_distances=True, clone_j=None, cloned_cells=None):
        self.edges = edges
        self.hs = hs
        self.vars_per_cell = vars_per_cell
        self.unit_cells_per_row = unit_cells_per_row
        self.graph_cache = graph_cache_dir
        self.sorted_edges = list(sorted(edges))
        self.edge_lookup = {edge: i for i, edge in enumerate(self.sorted_edges)}
        self.all_vars = list(sorted(set(v for edge in self.edges for v in edge)))
        self.periodic_boundaries = periodic_boundaries
        self.clone_j = clone_j
        self.cloned_cells = cloned_cells or {}

        if not self.load_if_needed() and calculate_traits:
            print("\tCalculating graph features")
            # Order matters
            self.unit_cells, self.unit_cell_bounding_box = self.calculate_unit_cells()
            self.used_unit_cells_per_row = max(max(a, b) for (a, b, _) in self.unit_cells) + 1
            self.dimer_vertex_list = self.calculate_dimer_vertex_list()
            self.edge_to_vertex_matrix = self.calculate_edge_to_vertex_matrix()
            if calculate_distances:
                self.vertex_distances, self.distance_lookup = variable_distances(self.sorted_edges)
                self.vertex_euclidean_distances = self.calculate_euclidean_distances(self.distance_lookup)
                self.dimer_distance_mat, self.all_dimer_pairs = self.calculate_dimer_distances()
                self.dimer_euclidean_distances = self.calculate_euclidean_dimer_distances()
                self.dimer_vertex_distances = self.calculate_dimer_vertex_euclidean_distance()
            self.save()
        elif calculate_traits:
            print("\tLoaded graph features")

    def var_is_in_cloned_cell(self, var):
        cx, cy, _ = get_var_traits(var)
        return (cx, cy) in self.cloned_cells

    def __hash__(self):
        return hash((
            tuple(self.sorted_edges),
            # tuple(self.hs.items()),
            self.vars_per_cell,
            self.unit_cells_per_row
        ))

    def load_if_needed(self):
        filename = os.path.join(self.graph_cache, '{}.pickle'.format(hash(self)))
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    config = pickle.load(f)
                return self.overwrite_with(config)
            except IOError as e:
                print("Error reading graph: {}".format(e))
        return False

    def overwrite_with(self, config):
        # self.edges = config.edges
        # self.hs = config.hs
        # self.vars_per_cell = config.vars_per_cell
        # self.unit_cells_per_row = config.unit_cells_per_row
        # self.graph_cache = config.graph_cache
        try:
            self.sorted_edges = config.sorted_edges
            self.edge_lookup = config.edge_lookup
            self.all_vars = config.all_vars
            self.unit_cells = config.unit_cells
            self.used_unit_cells_per_row = config.used_unit_cells_per_row
            self.unit_cell_bounding_box = config.unit_cell_bounding_box
            self.vertex_distances = config.vertex_distances
            self.distance_lookup = config.distance_lookup
            self.dimer_distance_mat = config.dimer_distance_mat
            self.all_dimer_pairs = config.all_dimer_pairs
            self.dimer_vertex_list = config.dimer_vertex_list
            self.edge_to_vertex_matrix = config.edge_to_vertex_matrix
            self.vertex_euclidean_distances = config.vertex_euclidean_distances
            self.dimer_euclidean_distances = config.dimer_euclidean_distances
            self.dimer_vertex_distances = config.dimer_vertex_distances
            self.clone_j = config.clone_j
            self.cloned_cells = config.cloned_cells
            return True
        except AttributeError as e:
            print(e)
            return False

    def save(self):
        if not os.path.exists(self.graph_cache):
            os.makedirs(self.graph_cache)
        filename = os.path.join(self.graph_cache, '{}.pickle'.format(hash(self)))
        with open(filename, 'wb') as w:
            pickle.dump(self, w)

    def non_cloned_edges(self):
        def is_cloned(edge):
            if self.clone_j is not None:
                return self.edges[edge] == self.clone_j
            return False
        return [edge for edge in self.sorted_edges if not is_cloned(edge)]

    def calculate_euclidean_distances(self, variables, inner_edge_d=1.0, output_edge_d=1.0):
        points = numpy.asarray([get_var_cartesian(var, self.vars_per_cell, self.unit_cells_per_row,
                                                  inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
                                for var in variables])
        return scipy.spatial.distance_matrix(points, points)

    def calculate_dimer_distances(self):
        edge_lookup = self.edge_lookup
        vertices_to_edges = collections.defaultdict(list)
        for va, vb in self.edge_lookup:
            vertex_a, vertex_b = get_dimer_vertices_for_edge(va, vb, vars_per_cell=self.vars_per_cell,
                                                             unit_cells_per_row=self.unit_cells_per_row,
                                                             used_unit_cells_per_row=self.used_unit_cells_per_row,
                                                             periodic_boundaries=self.periodic_boundaries)
            vertices_to_edges[vertex_a].append(edge_lookup[(va, vb)])
            vertices_to_edges[vertex_b].append(edge_lookup[(va, vb)])
        dimer_pairs = set()
        for _, edges in vertices_to_edges.items():
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    edge_pair = tuple(sorted([edges[i], edges[j]]))
                    dimer_pairs.add(edge_pair)
        return variable_distances(dimer_pairs)

    def calculate_euclidean_dimer_distances(self, inner_edge_d=1.0, output_edge_d=1.0):
        def average_pos(vara, varb):
            xa, ya = get_var_cartesian(vara, vars_per_cell=self.vars_per_cell,
                                       unit_cells_per_row=self.unit_cells_per_row,
                                       inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)

            xb, yb = get_var_cartesian(varb, vars_per_cell=self.vars_per_cell,
                                       unit_cells_per_row=self.unit_cells_per_row,
                                       inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
            return [(xa + xb) / 2.0, (ya + yb) / 2.0]

        points = numpy.asarray([average_pos(vara, varb) for vara, varb in self.sorted_edges])
        return scipy.spatial.distance_matrix(points, points)

    def calculate_unit_cells(self):
        """Return sorted list of all unit cells in graph. (x, y, is_front)"""
        var_traits = (get_var_traits(var_indx, self.vars_per_cell, self.unit_cells_per_row)
                      for var_indx in self.all_vars)
        unit_cells = list(sorted(set((cx, cy, is_front(indx, self.vars_per_cell))
                                     for cx, cy, indx in var_traits)))

        minx, miny = self.unit_cells_per_row, self.unit_cells_per_row
        maxx, maxy = 0, 0
        for cx, cy, _ in unit_cells:
            minx = min(minx, cx)
            miny = min(miny, cy)
            maxx = max(maxx, cx)
            maxy = max(maxy, cy)
        unit_cell_bounding_box = ((minx, miny), (maxx, maxy))
        return unit_cells, unit_cell_bounding_box

    def calculate_dimer_vertex_list(self):
        # TODO update this to deal with front/rear correctly.
        # First all unit cells
        unit_cells = set(self.unit_cells)
        # Then all the squares of unit cells
        unit_cycles = set()

        for (cx, cy, front) in unit_cells:
            def f(u):
                if self.periodic_boundaries:
                    # TODO make x and y different
                    return u % self.used_unit_cells_per_row
                else:
                    return u

            # Add the bottom-right one for each cell
            sides = [
                (cx, cy, front),
                (f(cx + 1), cy, front),
                (f(cx + 1), f(cy + 1), front),
                (cx, f(cy + 1), front)
            ]
            if all(side in unit_cells for side in sides[1:]):
                unit_cycles.add(tuple(sorted(tuple(sides))))
        dimer_vertex_list = []
        dimer_vertex_list.extend(sorted(unit_cells))
        dimer_vertex_list.extend(sorted(unit_cycles))
        return dimer_vertex_list

    def calculate_edge_to_vertex_matrix(self):
        dimer_vertex_list = self.dimer_vertex_list
        dimer_vertex_lookup = {k: i for i, k in enumerate(dimer_vertex_list)}
        edge_to_vertex_matrix = numpy.zeros((len(self.edges), len(dimer_vertex_list)), dtype=numpy.int8)
        for i,(va, vb) in enumerate(self.sorted_edges):
            v1, v2 = get_dimer_vertices_for_edge(va, vb, vars_per_cell=self.vars_per_cell,
                                                 unit_cells_per_row=self.unit_cells_per_row,
                                                 used_unit_cells_per_row=self.used_unit_cells_per_row,
                                                 periodic_boundaries=self.periodic_boundaries)
            if v1 in dimer_vertex_lookup:
                edge_to_vertex_matrix[i, dimer_vertex_lookup[v1]] = 1
            if v2 in dimer_vertex_lookup:
                edge_to_vertex_matrix[i, dimer_vertex_lookup[v2]] = 1
        return edge_to_vertex_matrix

    def calculate_dimer_vertex_euclidean_distance(self, inner_edge_d=1.0, output_edge_d=1.0):
        def dimer_vertex_position(dimer_vertex):
            # If it's a single unit cell (cx, cy, front)
            if len(dimer_vertex) == 3:
                dimer_vertex = [dimer_vertex]
            x, y = 0, 0
            for cx, cy, _ in dimer_vertex:
                dx, dy = get_unit_cell_cartesian(cx, cy, inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
                x += dx
                y += dy
            x = x / float(len(dimer_vertex))
            y = y / float(len(dimer_vertex))
            return numpy.asarray([x, y])

        points = numpy.asarray([dimer_vertex_position(dimer_vertex)
                                for dimer_vertex in self.dimer_vertex_list])
        return scipy.spatial.distance_matrix(points, points)


def is_front(index, vars_per_cell=8):
    """Returns if the absolute index is a front or rear unit cell."""
    var_relative = index % vars_per_cell
    return var_relative in [0, 1, 4, 5]


def is_unit_cell_type_a(unit_x, unit_y):
    """Returns if unit cell is of type A (true) or B (false)"""
    return ((unit_x + unit_y) % 2) == 0


def calculate_variable_direction(var_index, vars_per_cell=8, unit_cells_per_row=16):
    """Calculate the direction associated with the variable."""
    unit_x, unit_y, rel_var = get_var_traits(var_index, vars_per_cell=vars_per_cell,
                                             unit_cells_per_row=unit_cells_per_row)
    dx, dy, side = GraphBuilder.var_connections[rel_var]
    # dx and dy are defined as A to B, so if B then reverse
    if not is_unit_cell_type_a(unit_x, unit_y):
        dx, dy = -dx, -dy
    return dx, dy


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


def get_abs_var_traits(index, vars_per_cell=8, unit_cells_per_row=16):
    cx, cy, rel_var = get_var_traits(index, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    lat_var, front = LatticeVar.get_from_rel_var(cx, cy, rel_var)
    return cx, cy, lat_var, front


def relative_pos_of_relative_index(relative_index, dist=1.0):
    """
    Get the relative position of a relative index compared to its unit cell center.
    :param relative_index:
    :param dist:
    :return:
    """
    if relative_index == 0 or relative_index == 2:
        return 0, -dist
    elif relative_index == 1 or relative_index == 3:
        return 0, dist
    elif relative_index == 4 or relative_index == 6:
        return -dist, 0
    elif relative_index == 5 or relative_index == 7:
        return dist, 0


def get_unit_cell_cartesian(unit_x, unit_y, inner_edge_d=1.0, output_edge_d=1.0):
    """Get the cartesian position of a unit cell."""
    # sqrt(d^2 + d^2)/2 = sqrt(2 d^2)/2 = sqrt(d^2 / 2) = d/sqrt(2)
    unit_cell_radius = inner_edge_d / numpy.sqrt(2)
    # One radius on either side plus bond between them.
    unit_cell_d = 2 * unit_cell_radius + output_edge_d
    cx, cy = unit_cell_d * unit_x, unit_cell_d * unit_y
    return cx, cy


def get_var_cartesian(index, vars_per_cell=8, unit_cells_per_row=16, inner_edge_d=1.0, output_edge_d=1.0):
    """
    Get the cartesian coords of the variable.
    :param index:
    :param vars_per_cell:
    :param unit_cells_per_row:
    :param inner_edge_d:
    :param output_edge_d:
    :return:
    """
    unit_x, unit_y, rel_indx = get_var_traits(index, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    # sqrt(d^2 + d^2)/2 = sqrt(2 d^2)/2 = sqrt(d^2 / 2) = d/sqrt(2)
    unit_cell_radius = inner_edge_d / numpy.sqrt(2)
    cx, cy = get_unit_cell_cartesian(unit_x, unit_y, inner_edge_d=inner_edge_d, output_edge_d=output_edge_d)
    dx, dy = relative_pos_of_relative_index(rel_indx, unit_cell_radius)
    if not is_type_a(unit_x, unit_y):
        dx = -dx
        dy = -dy
    return cx + dx, cy + dy


def get_variable_for_cell_dir(unit_x, unit_y, dx, dy, vars_per_cell=8, unit_cells_per_row=16, front=True):
    if not is_type_a(unit_x, unit_y):
        dx = -dx
        dy = -dy

    unit_cell_index = unit_y * unit_cells_per_row + unit_x
    rel_var = get_connection_cells()[(dx, dy, front)]

    return unit_cell_index*vars_per_cell + rel_var


def get_dimer_vertices_for_edge(vara, varb, vars_per_cell=8, unit_cells_per_row=16, used_unit_cells_per_row=16, periodic_boundaries=False):
    """Returns the two dimer vertices attached by this dimer."""
    acx, acy, rela = get_var_traits(vara, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    bcx, bcy, relb = get_var_traits(varb, vars_per_cell=vars_per_cell, unit_cells_per_row=unit_cells_per_row)
    front = is_front(vara)

    def f(u):
        if periodic_boundaries:
            # TODO make x and y different
            return u % used_unit_cells_per_row
        else:
            return u

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
            (f(cx + adx), f(cy + ady), front),
            (f(cx + bdx), f(cy + bdy), front),
            (f(cx + adx + bdx), f(cy + ady + bdy), front)
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
            (f(acx + dy), f(acy + dx), front),
            (f(bcx + dy), f(bcy + dx), front),
        ]))
        unit_cycle_b = tuple(sorted([
            (acx, acy, front),
            (bcx, bcy, front),
            (f(acx - dy), f(acy - dx), front),
            (f(bcx - dy), f(bcy - dx), front),
        ]))
        return unit_cycle_a, unit_cycle_b


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


def make_periodic_connection(place_in_cell, connecting_sides_of_cell, bond_j, clone_j, vars_per_cell=8, ideal_clones=False):
    cx, cy = place_in_cell
    ux, uy = connecting_sides_of_cell

    dx, dy = ux - cx, uy - cy
    assert abs(dx) + abs(dy) == 1

    if is_type_a(ux, uy):
        dx = -dx
        dy = -dy
    conns = get_connection_cells()
    front_conn = conns[(dx, dy, True)]
    front_clone = get_var_across(front_conn)

    # Vertical is identical, horizontal is flipped
    if abs(dx) == 1 and abs(dy) == 0:
        dx = -dx
        dy = -dy

    back_conn = conns[(dx, dy, False)]
    back_clone = get_var_across(back_conn)

    front_from = var_num(ux, uy, front_conn, vars_per_cell=vars_per_cell)
    front_to = var_num(cx, cy, front_conn, vars_per_cell=vars_per_cell)
    front_to_clone = var_num(cx, cy, front_clone, vars_per_cell=vars_per_cell)

    back_from = var_num(ux, uy, back_conn, vars_per_cell=vars_per_cell)
    back_to = var_num(cx, cy, back_conn, vars_per_cell=vars_per_cell)
    back_to_clone = var_num(cx, cy, back_clone, vars_per_cell=vars_per_cell)

    if ideal_clones:
        connections = [
            ((front_from, back_from), bond_j)
        ]
    else:
        connections = [
            # Clones
            ((front_from, front_to), clone_j),
            ((back_from, back_to), clone_j),
            ((front_to, front_to_clone), clone_j),
            ((back_to, back_to_clone), clone_j),
            # Bonds
            ((front_to, back_to_clone), bond_j),
            ((back_to, front_to_clone), bond_j)
        ]

    return {(min(a, b), max(a, b)): j for ((a, b), j) in connections}


def get_var_across(cell_var):
    return (cell_var + 4) % 8


def is_type_a(x, y):
    return ((x + y) % 2) == 0


def chimera_index(unit_cell, var_index, vars_per_cell=8):
    return (unit_cell * vars_per_cell) + var_index


def var_num(unit_x, unit_y, var_unit, vars_per_cell=8, units_per_row=16):
    unit_num = unit_x + unit_y * units_per_row
    return chimera_index(unit_num, var_unit, vars_per_cell=vars_per_cell)


def convert_for_cell(unit_x, unit_y, var_dict):
    def f(x):
        return var_num(unit_x, unit_y, x)

    return {(f(x), f(y)): v for (x, y), v in var_dict.items()}


def make_unit_cell_a(unit_x, unit_y, j=1.0, front=True):
    # Internal edges
    bond = -j
    if front:
        return convert_for_cell(unit_x, unit_y, {
            (0, 4): bond,
            (1, 4): bond,
            (1, 5): bond,
            (0, 5): -bond,
        })
    else:
        # Above but +2
        return convert_for_cell(unit_x, unit_y, {
            (2, 6): bond,
            (3, 6): bond,
            (3, 7): bond,
            (2, 7): -bond,
        })


def make_unit_cell_b(unit_x, unit_y, j=1.0, front=True):
    # 4 <-> 5 and 0 <-> 1
    bond = -j
    if front:
        return convert_for_cell(unit_x, unit_y, {
            (0, 4): bond,
            (1, 4): -bond,
            (1, 5): bond,
            (0, 5): bond,
        })
    else:
        # Above but +2
        return convert_for_cell(unit_x, unit_y, {
            (2, 6): bond,
            (3, 6): -bond,
            (3, 7): bond,
            (2, 7): bond,
        })


def make_configs(all_vars):
    num_vars = len(all_vars)
    for config in range(1 << num_vars):
        yield {
            v: ((config >> i) & 1) * 2 - 1
            for i, v in enumerate(all_vars)
        }


def energy_of_bonds(bonds, config):
    return sum(v * config[a] * config[b] for (a, b), v in bonds.items())


def energy_states(bonds):
    all_vars = list(sorted(set(x for k in bonds for x in k)))
    for var_config in make_configs(all_vars):
        yield var_config, energy_of_bonds(bonds, var_config)
