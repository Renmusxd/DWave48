from bathroom_tile import graphbuilder
import py_monte_carlo
import numpy
import collections


class QuantumMonteCarloSampler:
    def __init__(self, beta=39.72, thermalization_time=1e6, timesteps=2e6, sampling_freq=1e3):
        self.beta = beta
        self.timesteps = int(timesteps)
        self.wait_time = int(thermalization_time)
        self.sampling_freq = int(sampling_freq)

    def sample_ising(self, hs, edges, num_reads=1, transverse_field=1.0, auto_scale=True):
        assert all(h == 0 for h in hs)
        all_vars = list(sorted(set(v for (va, vb) in edges for v in [va, vb])))
        all_vars_lookup = {v: i for i, v in enumerate(all_vars)}

        max_abs_e = 1.0
        if auto_scale:
            min_j = min(j for _, j in edges.items())
            max_j = max(j for _, j in edges.items())
            max_e = max(transverse_field, max_j)
            min_e = min(transverse_field, min_j)
            max_abs_e = max(abs(max_e), abs(min_e))

        edges = [((all_vars_lookup[va], all_vars_lookup[vb]), j / max_abs_e) for (va, vb), j in edges.items()]
        transverse_field = transverse_field / max_abs_e

        effective_timesteps = self.timesteps - self.wait_time
        samples_per_experiment = effective_timesteps / self.sampling_freq
        experiments = int(numpy.ceil(num_reads / samples_per_experiment))

        print("(Running {} independent experiments each making {} samples)".format(experiments, samples_per_experiment))

        lattice = py_monte_carlo.Lattice(edges)
        lattice.set_transverse_field(transverse_field)

        energies, states = lattice.run_quantum_monte_carlo_sampling(self.beta, self.timesteps, experiments,
                                                                    sampling_wait_buffer=self.wait_time,
                                                                    sampling_freq=self.sampling_freq,
                                                                    use_loop_update=False,
                                                                    use_heatbath_diagonal_update=False)

        # Flatten except variables
        states = states.reshape((-1, states.shape[-1]))
        states = states * 2 - 1
        average_energy = numpy.mean(energies)
        edges_dict = dict(edges)
        energies = [graphbuilder.energy_of_bonds(edges_dict, sample) for sample in states]

        return QuantumMonteCarloResponse(states, energies, average_actual_energy=average_energy)


class QuantumMonteCarloResponse:
    def __init__(self, data, energies, average_actual_energy=None):
        self.my_data = data.T
        self.energies = energies
        self.average_actual_energy = average_actual_energy

    def data(self):
        return self.my_data

    def energy(self):
        return self.energies

    def average_energy(self):
        return self.average_actual_energy

    def scalars(self):
        return {'average_energy': self.average_actual_energy}
