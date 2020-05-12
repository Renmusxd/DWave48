from dwave.system import samplers


class DWaveSampler:
    """Wrapper for samplers.DWaveSampler"""
    def __init__(self):
        pass

    def sample_ising(self, hs, edges, num_reads=1, auto_scale=True, transverse_field=0.0):
        machine_sampler = samplers.DWaveSampler()
        return machine_sampler.sample_ising(hs, edges, num_reads=num_reads, auto_scale=auto_scale)
