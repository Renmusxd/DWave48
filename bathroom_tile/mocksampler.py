from bathroom_tile.graphbuilder import energy_of_bonds, get_var_traits


class MockSampler:
    def sample_ising(self, hs, edges, num_reads=1, auto_scale=True):
        variables = set(v for edge in edges for v in edge)

        data = []
        for fn in [top_left, top_right, bot_left, bot_right]:
            sample = MockSample(variables, sampling_fn=fn)
            energy = energy_of_bonds(edges, {k: sample[k] for k in variables})
            data.append((sample, energy, num_reads))
        return MockResponse(data)


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
