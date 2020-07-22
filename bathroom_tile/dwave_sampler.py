from dwave.system import samplers
from bathroom_tile.mocksampler import columnar
import scipy.interpolate
import numpy

CLIENT_SINGLETON = None


def get_client_singleton():
    global CLIENT_SINGLETON
    if CLIENT_SINGLETON is None:
        CLIENT_SINGLETON = samplers.DWaveSampler()
    return CLIENT_SINGLETON


class MachineTransverseFieldHelper:
    def __init__(self, s_map_file='dwave_energies/dw_2000q_2_1.txt'):
        with open(s_map_file, "r") as f:
            ss, a_s, b_s = [], [], []
            for line in f:
                words = line.split()
                ss.append(float(words[0]))
                a_s.append(float(words[1]))
                b_s.append(float(words[2]))
            bmax = b_s[-1]
            s_map = list(zip(ss, (a/bmax for a in a_s)))
            s_bmap = list(zip(ss, (b/bmax for b in b_s)))

        a_xs, a_ys = zip(*s_map)
        self.max_field = a_ys[0]
        self.s_to_a = scipy.interpolate.interp1d(a_xs, a_ys)
        self.a_to_s = scipy.interpolate.interp1d(a_ys, a_xs)

        b_xs, b_ys = zip(*s_bmap)
        self.s_to_b = scipy.interpolate.interp1d(b_xs, b_ys)
        self.b_to_s = scipy.interpolate.interp1d(b_ys, b_xs)

        ratios = list(a/b for a, b in zip(a_ys, b_ys))
        self.s_to_a_over_b = scipy.interpolate.interp1d(b_xs, ratios)
        self.a_over_b_to_s = scipy.interpolate.interp1d(ratios, b_xs)

        self.s_map = s_map
        self.s_bmad = s_bmap

    def s_for_gamma(self, gamma):
        return float(self.a_to_s(gamma))

    def s_for_gamma_ratio(self, gamma_over_j):
        return float(self.a_over_b_to_s(gamma_over_j))

    def gamma_for_s(self, s):
        return float(self.s_to_a(s))

    def s_for_j(self, j):
        return float(self.b_to_s(j))

    def j_for_s(self, s):
        return float(self.s_to_b(s))


class DWaveSampler:
    """Wrapper for samplers.DWaveSampler"""
    def __init__(self, ramp_up_slope=0.05, hold_time=50, quench_slope=1, s_map_file='dwave_energies/dw_2000q_2_1.txt', initial_state=None, dry_run=False):
        """
        Make a new DWaveSampler.
        :param ramp_up_slope:
        :param hold_time: hold time at transverse_field.
        :param quench_slope: slope to use for quench
        :param s_map: [..., (s, A(s)/Bmax), ...]
        """
        self.ramp_slope = ramp_up_slope
        self.hold_time = hold_time
        self.quench_slope = quench_slope
        self.field_helper = MachineTransverseFieldHelper(s_map_file=s_map_file)
        self.initial_state = initial_state
        self.dry_run = dry_run

    def sample_ising(self, hs, edges, num_reads=1, auto_scale=True, transverse_field=0.0):
        if self.initial_state:
            raise NotImplementedError()

        schedule = None
        if not 0.0 <= transverse_field <= self.field_helper.max_field:
            raise Exception("Transverse field must be between 0.0 and {:0.3f}".format(self.field_helper.max_field))
        elif transverse_field > 0.0:
            s = self.field_helper.s_for_gamma(transverse_field)
            t = s/self.ramp_slope
            t_quench = (1.0 - s)/self.quench_slope
            schedule = [(0, 0.0),
                        (t, s), (t+self.hold_time, s),
                        (t+self.hold_time+t_quench, 1.0)]
        else:
            s = 1.0

        all_vars = numpy.asarray(list(sorted(set([v for edge in edges for v in [edge[0], edge[1]]]))))

        j_ratio = self.field_helper.j_for_s(s)
        edge_j = max(edges.values())
        gamma_ratio = transverse_field / (edge_j * j_ratio)

        print("Sampling with:")
        print("\ts:\t{}".format(s))
        print("\tj:\t{}".format(edge_j))
        print("\tB(s):\t{}".format(j_ratio))
        print("\tg:\t{}".format(transverse_field))
        print("\tg/(j B(s)):\t{}".format(gamma_ratio))

        if not self.dry_run:
            machine_sampler = get_client_singleton()
            dwave_samples = machine_sampler.sample_ising(hs, edges, num_reads=num_reads, auto_scale=auto_scale,
                                                         anneal_schedule=schedule)
            energies = [energy for _, energy, num_occ in dwave_samples.data() for _ in range(num_occ)]
            data = [[sample[k] for k in all_vars] for sample, _, num_occ in dwave_samples.data() for _ in range(num_occ)]
            data = numpy.asarray(data).T

            return DWaveResponse(data, energies, all_vars, s=s, j=j_ratio, gamma=transverse_field)
        elif self.dry_run == "columnar" or (type(self.dry_run) == float and self.dry_run > gamma_ratio):
            data = numpy.expand_dims([columnar(v) for v in all_vars], axis=-1)
            data = numpy.tile(data, (1,num_reads))
            energies = numpy.zeros((num_reads,))
            return DWaveResponse(data, energies, all_vars, s=s, j=j_ratio, gamma=transverse_field)
        else:
            data = numpy.random.choice(a=[-1, 1], size=(len(all_vars), num_reads), p=[0.5, 0.5])
            energies = numpy.zeros((num_reads,))
            return DWaveResponse(data, energies, all_vars, s=s, j=j_ratio, gamma=transverse_field)


class DWaveResponse:
    def __init__(self, data, energies, all_vars, s, j, gamma):
        self.energies = energies
        self.my_data = data
        self.s = s
        self.j = j
        self.gamma = gamma
        self.all_vars = all_vars

    def var_map(self):
        return self.all_vars

    def data(self):
        return self.my_data

    def energy(self):
        return self.energies

    def scalars(self):
        return {"actual_j": self.j, "s": self.s}