import py_monte_carlo
import collections


class MonteCarloSampler:
    def __init__(self, beta=39.72, timesteps=1e6, annealed=True, only_single_spin_flips=False):
        self.beta = beta
        self.timesteps = timesteps
        self.annealed = annealed
        self.basic_moves = only_single_spin_flips

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
        if self.annealed:
            es, ss = lattice.run_monte_carlo_annealing(annealing, t, num_reads, only_basic_moves=self.basic_moves)
        else:
            es, ss = lattice.run_monte_carlo(self.beta, t, num_reads, only_basic_moves=self.basic_moves)

        ss = ss*2 - 1

        return MonteCarloResponse(ss, es)


class MonteCarloResponse:
    def __init__(self, data, energies):
        self.my_data = data.T
        self.energies = energies

    def data(self):
        return self.my_data

    def energy(self):
        return self.energies

    def scalars(self):
        return {}