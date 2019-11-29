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


class Graph:
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
            if (dx, dy, front) in Graph.connection_cells:
                v = Graph.connection_cells[(dx, dy, front)]
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

        for loc, adj in self.periodic_boundaries:
            if is_type_a(*loc):
                cella, cellb = loc, adj
            else:
                cella, cellb = adj, loc
            dx = cellb[0] - cella[0]
            dy = cellb[1] - cella[1]
            v_front = Graph.connection_cells[(dx, dy, True)]
            v_rear = Graph.connection_cells[(dx, dy, False)]
            clone_edges = make_boundary_cell(loc, adj, v_front, v_rear, -self.j, -self.clone)
            connections.update(clone_edges)
            for (va, vb), j in clone_edges.items():
                raise NotImplementedError("Have not implemented periodic boundary conditions with transverse field.")

        return hs, connections

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
            for dx, dy, conn_side in Graph.connection_cells:
                if front != conn_side:
                    continue
                ox = x + dx
                oy = y + dy
                if (ox, oy, front) in self.unit_cells:
                    self.connect_cells((x, y), (ox, oy), front=front)


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
        yield (var_config, energy_of_bonds(bonds, var_config))
