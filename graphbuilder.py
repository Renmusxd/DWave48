class MockSample:
    def __getitem__(self, index):
        if index == 5:
            return 0
        if index == 13:
            return 1
        return 0


class Graph:
    def __init__(self, j=1.0):
        self.unit_cells = set()
        self.j = j
        self.connection_cells = {
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
        self.existing_cell_connections = set()

    def build(self):
        connections = {}
        for x, y, front in self.unit_cells:
            if ((x + y) % 2) == 0:
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
            if (dx, dy, front) in self.connection_cells:
                v = self.connection_cells[(dx, dy, front)]
                abs_va = var_num(cella[0], cella[1], v)
                abs_vb = var_num(cellb[0], cellb[1], v)
                # Enter them in sorted order
                min_v = min(abs_va, abs_vb)
                max_v = max(abs_va, abs_vb)
                edge = (min_v, max_v)
                connections.update({edge: -self.j})
        return connections

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
            for dx, dy, conn_side in self.connection_cells:
                if front != conn_side:
                    continue
                ox = x + dx
                oy = y + dy
                if (ox, oy, front) in self.unit_cells:
                    self.connect_cells((x, y), (ox, oy), front=front)


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


def make_graph_with_unit_cells(unit_cells, j=1.0):
    unit_cells = set(unit_cells)
    indiv_cell_graphs = [
        make_unit_cell_a(x, y, j=j) if ((x+y) % 2) == 0 else make_unit_cell_b(x, y, j=j)
        for x, y in unit_cells
    ]
    full_dict = {}
    for d in indiv_cell_graphs:
        full_dict.update(d)

    deltas_and_vars = {
        (0, -1): 0,
        (0, 1): 1,
        (-1, 0): 4,
        (1, 0): 5,
    }

    # Now add connections
    for x, y in unit_cells:
        # type is false (A) or true (B)
        own_type = ((x+y) % 2) == 1

        for (dx, dy), v in deltas_and_vars.items():
            if own_type:
                ox, oy = x - dx, y - dy
            else:
                ox, oy = x + dx, y + dy
            if (ox, oy) in unit_cells:
                # For B sublattice, swap nodes
                abs_va = var_num(x, y, v)
                abs_vb = var_num(ox, oy, v)
                # Enter them in sorted order
                min_v = min(abs_va, abs_vb)
                max_v = max(abs_va, abs_vb)
                edge = (min_v, max_v)
                full_dict.update({edge: -j})
    return full_dict


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
