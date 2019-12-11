import numpy
import collections
import pickle
import os


class MockSample:
    def __getitem__(self, index):
        if index == 5:
            return 0
        if index == 13:
            return 1
        return 0


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


class GraphBuilder:
    connection_cells = get_connection_cells()
    var_connections = get_var_to_connection_map()

    def __init__(self, j=1.0, clone_strength=5.0):
        self.unit_cells = set()
        self.j = j
        self.clone = clone_strength
        self.existing_cell_connections = set()
        self.periodic_boundaries = set()

    def add_periodic_boundary(self, loc, adj):
        if loc in self.unit_cells:
            raise Exception("Attempting to superimpose boundary and cell: {}".format(loc))
        self.periodic_boundaries.add((loc, adj))

    def build(self, h=0):
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

        # TODO: fix
        # for loc, adj in self.periodic_boundaries:
        #     if is_type_a(*loc):
        #         cella, cellb = loc, adj
        #     else:
        #         cella, cellb = adj, loc
        #     dx = cellb[0] - cella[0]
        #     dy = cellb[1] - cella[1]
        #     v_front = GraphBuilder.connection_cells[(dx, dy, True)]
        #     v_rear = GraphBuilder.connection_cells[(dx, dy, False)]
        #     clone_edges = make_boundary_cell(loc, adj, v_front, v_rear, -self.j, -self.clone)
        #     connections.update(clone_edges)
        #     for (va, vb), j in clone_edges.items():
        #         raise NotImplementedError("Have not implemented periodic boundary conditions with transverse field.")

        return Graph(connections, hs)

    def add_cells(self, cells, fronts=None):
        if fronts is None:
            fronts = (True for _ in range(len(cells)))
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
    def __init__(self, edges, hs, graph_cache_dir='graphcache', vars_per_cell=8, unit_cells_per_row=16):
        self.edges = edges
        self.hs = hs
        self.vars_per_cell = vars_per_cell
        self.unit_cells_per_row = unit_cells_per_row
        self.graph_cache = graph_cache_dir
        self.sorted_edges = list(sorted(edges))
        self.edge_lookup = {edge: i for i, edge in enumerate(self.sorted_edges)}
        self.all_vars = list(sorted(set(v for edge in self.edges for v in edge)))

        if not self.load_if_needed():
            # Order matters
            self.unit_cells, self.unit_cell_bounding_box = self.calculate_unit_cells()
            self.vertex_distances, self.distance_lookup = variable_distances(self.sorted_edges)
            self.dimer_distance_mat, self.all_dimer_pairs = self.calculate_dimer_distances()
            self.dimer_vertex_list = self.calculate_dimer_vertex_list()
            self.edge_to_vertex_matrix = self.calculate_edge_to_vertex_matrix()
            self.save()

    def __hash__(self):
        return hash((
            tuple(self.sorted_edges),
            tuple(self.hs.items()),
            self.vars_per_cell,
            self.unit_cells_per_row
        ))

    def load_if_needed(self):
        filename = os.path.join(self.graph_cache, '{}.pickle'.format(hash(self)))
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                config = pickle.load(f)
            self.overwrite_with(config)
            return True
        return False

    def overwrite_with(self, config):
        self.edges = config.edges
        self.hs = config.hs
        self.vars_per_cell = config.vars_per_cell
        self.unit_cells_per_row = config.unit_cells_per_row
        self.graph_cache = config.graph_cache
        self.sorted_edges = config.sorted_edges
        self.edge_lookup = config.edge_lookup
        self.all_vars = config.all_vars
        self.unit_cells = config.unit_cells
        self.unit_cell_bounding_box = config.unit_cell_bounding_box
        self.vertex_distances = config.vertex_distances
        self.distance_lookup = config.distance_lookup
        self.dimer_distance_mat = config.dimer_distance_mat
        self.all_dimer_pairs = config.all_dimer_pairs
        self.dimer_vertex_list = config.dimer_vertex_list
        self.edge_to_vertex_matrix = config.edge_to_vertex_matrix

    def save(self):
        if not os.path.exists(self.graph_cache):
            os.makedirs(self.graph_cache)
        filename = os.path.join(self.graph_cache, '{}.pickle'.format(hash(self)))
        with open(filename, 'wb') as w:
            pickle.dump(self, w)

    def calculate_dimer_distances(self):
        edge_lookup = self.edge_lookup
        vertices_to_edges = collections.defaultdict(list)
        for va, vb in self.edge_lookup:
            vertex_a, vertex_b = get_dimer_vertices_for_edge(va, vb, vars_per_cell=self.vars_per_cell,
                                                             unit_cells_per_row=self.unit_cells_per_row)
            vertices_to_edges[vertex_a].append(edge_lookup[(va, vb)])
            vertices_to_edges[vertex_b].append(edge_lookup[(va, vb)])
        dimer_pairs = set()
        for _, edges in vertices_to_edges.items():
            for i in range(len(edges)):
                for j in range(i+1, len(edges)):
                    edge_pair = tuple(sorted([edges[i], edges[j]]))
                    dimer_pairs.add(edge_pair)
        return variable_distances(dimer_pairs)

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
            # Add the bottom-right one for each cell
            sides = [
                (cx, cy, front),
                (cx + 1, cy, front),
                (cx + 1, cy + 1, front),
                (cx, cy + 1, front)
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
        for (va, vb) in self.sorted_edges:
            v1, v2 = get_dimer_vertices_for_edge(va, vb, vars_per_cell=self.vars_per_cell,
                                                 unit_cells_per_row=self.unit_cells_per_row)
            i = self.edge_lookup[(va, vb)]
            if v1 in dimer_vertex_lookup:
                edge_to_vertex_matrix[i, dimer_vertex_lookup[v1]] = 1
            if v2 in dimer_vertex_lookup:
                edge_to_vertex_matrix[i, dimer_vertex_lookup[v2]] = 1
        return edge_to_vertex_matrix


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


def make_boundary_cell(boundary_cell, connected_cell, v_front, v_rear, bond_j, clone_j, vars_per_cell=8):
    def make_edge(c1, c2, v1, v2):
        abs_va = var_num(c1[0], c1[1], v1)
        abs_vb = var_num(c2[0], c2[1], v2)
        # Enter them in sorted order
        min_v = min(abs_va, abs_vb)
        max_v = max(abs_va, abs_vb)
        return min_v, max_v

    def make_edge_for_v(c1, c2, v):
        return make_edge(c1, c2, v, v)

    # variable on opposite side of unit cell from front
    v_intermediate = (v_front + vars_per_cell//2) % vars_per_cell

    front_clone_edge = make_edge_for_v(boundary_cell, connected_cell, v_front)
    extra_clone_edge = make_edge(boundary_cell, boundary_cell, v_front, v_intermediate)
    front_rear_edge = make_edge(boundary_cell, boundary_cell, v_intermediate, v_rear)
    rear_clone_edge = make_edge_for_v(boundary_cell, connected_cell, v_rear)

    return {
        front_clone_edge: clone_j,
        extra_clone_edge: clone_j,
        front_rear_edge: bond_j,
        rear_clone_edge: clone_j
    }


def is_type_a(x, y):
    return ((x + y) % 2) == 0


def chimera_index(unit_cell, var_index):
    vars_per_cell = 8
    return (unit_cell * vars_per_cell) + var_index


def var_num(unit_x, unit_y, var_unit):
    units_per_row = 16
    unit_num = unit_x + unit_y * units_per_row
    return chimera_index(unit_num, var_unit)


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
            (2, 6): -bond,
            (3, 6): bond,
            (3, 7): bond,
            (2, 7): bond,
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
            (3, 6): bond,
            (3, 7): -bond,
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
    return sum(v*config[a]*config[b] for (a, b), v in bonds.items())


def energy_states(bonds):
    all_vars = list(sorted(set(x for k in bonds for x in k)))
    for var_config in make_configs(all_vars):
        yield var_config, energy_of_bonds(bonds, var_config)
