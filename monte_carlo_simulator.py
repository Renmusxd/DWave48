import py_monte_carlo
import collections


class MonteCarloSampler:
    def __init__(self, beta=39.72, timesteps=1e6, annealed=True, only_single_spin_flips=False, read_all_energies=False):
        self.beta = beta
        self.timesteps = timesteps
        self.annealed = annealed
        self.basic_moves = only_single_spin_flips
        self.read_all_energies = read_all_energies

    def sample_ising(self, hs, edges, num_reads=1, auto_scale=True):
        all_vars = list(sorted(set(v for (va, vb) in edges for v in [va, vb])))
        all_vars_lookup = {v: i for i, v in enumerate(all_vars)}

        max_abs_e = 1.0
        if auto_scale:
            min_h = min(h for _, h in hs.items())
            max_h = max(h for _, h in hs.items())
            min_j = min(j for _, j in edges.items())
            max_j = max(j for _, j in edges.items())
            max_e = max(max_h, max_j)
            min_e = min(min_h, min_j)
            max_abs_e = max(abs(max_e), abs(min_e))
        for v in all_vars:
            if v not in hs:
                hs[v] = 0.0
        edges = [((all_vars_lookup[va], all_vars_lookup[vb]), j / max_abs_e) for (va, vb), j in edges.items()]
        hs = [hs[v] / max_abs_e for v in all_vars]
        t = int(self.timesteps)
        annealing = [(0, 0.0), (t, self.beta)] if self.annealed else [(0, self.beta), (0, 0.0)]
        all_energies = None

        lattice = py_monte_carlo.Lattice(len(all_vars), edges)
        for i, h in enumerate(hs):
            lattice.set_bias(i, h)

        # TODO: verify the changes here since switching to numpy code
        if self.read_all_energies:
            all_energies, ss = lattice.run_monte_carlo_annealing_and_get_energies(annealing, t, num_reads, only_basic_moves=self.basic_moves)
            readout = ((energies[-1], s) for energies, s in zip(all_energies, ss))
        elif self.annealed:
            es, ss = lattice.run_monte_carlo_annealing(annealing, t, num_reads, only_basic_moves=self.basic_moves)
            readout = zip(es, ss)
        else:
            es, ss = lattice.run_monte_carlo(self.beta, t, num_reads, only_basic_moves=self.basic_moves)
            readout = zip(es, ss)
        readout = [(energy, tuple(s)) for energy, s in readout]
        num_occurences = collections.defaultdict(lambda: 0)
        for energy, s in readout:
            num_occurences[s] += 1

        def f(v):
            if v:
                return 1
            else:
                return -1

        readout = [({all_vars[i]: f(v) for i, v in enumerate(s)}, energy, num_occurences[s]) for energy, s in readout]
        return MonteCarloResponse(readout, all_energies=all_energies)


class MonteCarloResponse:
    def __init__(self, data, all_energies=None):
        self.my_data = data
        self.all_energies = all_energies

    def data(self):
        return self.my_data

    def trace_energies(self):
        return self.all_energies