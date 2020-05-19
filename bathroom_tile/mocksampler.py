from bathroom_tile.graphbuilder import energy_of_bonds, get_var_traits, is_unit_cell_type_a


class MockSampler:
    def sample_ising(self, hs, edges, num_reads=1, auto_scale=True, transverse_field=None):
        variables = set(v for edge in edges for v in edge)

        sample = MockSample(variables, sampling_fn=columnar)
        energy = energy_of_bonds(edges, {k: sample[k] for k in variables})
        data = (sample, energy, num_reads)
        return MockResponse([data])


class MockResponse:
    def __init__(self, data):
        self.my_data = data

    def data(self):
        return self.my_data


class MockSample:
    def __init__(self, variables, sampling_fn=None):
        self.variables = variables
        self.sampling_fn = sampling_fn

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, index):
        return self.sampling_fn(index)

def columnar(v):
    x, y, r = get_var_traits(v)
    if is_unit_cell_type_a(x, y):
        return 1 if (r % 2 == 0) == ((x + y) % 4 == 0) else -1
    else:
        return 1 if (x + y) % 4 != 1 else -1

def bot_left(index):
    _, _, rel = get_var_traits(index)
    if rel == 0:
        return 1
    elif rel == 1:
        return -1
    elif rel == 4:
        return 1
    elif rel == 5:
        return -1


def top_left(index):
    _, _, rel = get_var_traits(index)
    if rel == 0:
        return 1
    elif rel == 1:
        return -1
    elif rel == 4:
        return -1
    elif rel == 5:
        return -1


def top_right(index):
    return 1


def bot_right(index):
    _, _, rel = get_var_traits(index)
    if rel == 0:
        return 1
    elif rel == 1:
        return 1
    elif rel == 4:
        return 1
    elif rel == 5:
        return -1
