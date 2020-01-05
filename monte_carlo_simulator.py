import monte_carlo
import collections


class MonteCarloSampler:
    def __init__(self, beta=39.72, timesteps=1e7, annealed=True, only_single_spin_flips=False):
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
        edges = [((all_vars_lookup[va], all_vars_lookup[vb]), j/max_abs_e) for (va, vb), j in edges.items()]
        hs = [hs[v]/max_abs_e for v in all_vars]
        t = int(self.timesteps)
        if self.annealed:
            readout = monte_carlo.run_monte_carlo_annealing([(0, 0.0), (t, self.beta)],
                                                            t, num_reads, edges, hs, self.basic_moves)
        else:
            readout = monte_carlo.run_monte_carlo(self.beta, t, num_reads, edges, hs, self.basic_moves)
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
        return MonteCarloResponse(readout)


class MonteCarloResponse:
    def __init__(self, data):
        self.my_data = data

    def data(self):
        return self.my_data