import bathroom_tile
import numpy
import pickle
import os


class BathroomTileExperiment:
    def __init__(self, sampler, unit_cell_rect=None, j=1.0, gamma=0.0, graph=None, num_reads=10000, graph_build_kwargs=None, sampler_build_kwargs=None, sampler_sample_kwargs=None):
        if unit_cell_rect:
            (min_x, max_x), (min_y, max_y) = unit_cell_rect
            gb = bathroom_tile.graphbuilder.GraphBuilder(j=j)
            gb.add_cells([
                (x, y)
                for x in range(min_x, max_x)
                for y in range(min_y, max_y)
            ])
            gb.connect_all()
            graph_build_kwargs = graph_build_kwargs or {}
            self.graph = gb.build(calculate_traits=False, calculate_distances=False, **graph_build_kwargs)
        elif graph:
            self.graph = graph
        else:
            raise ValueError("graph or unit_cell_rect must be supplied")
        self.gamma = gamma
        self.sampler = sampler
        self.sampler_build_kwargs = sampler_build_kwargs
        self.sampler_sample_kwargs = sampler_sample_kwargs
        self.num_reads = num_reads

    def save_experiment_config(self, filepath):
        with open(filepath, 'wb') as w:
            obj = (self.graph, self.gamma, self.sampler, self.num_reads, (self.sampler_build_kwargs, self.sampler_sample_kwargs))
            pickle.dump(obj, w)

    @staticmethod
    def load_experiment_config(filepath):
        with open(filepath, 'rb') as f:
            graph, gamma, sampler, num_reads, (sampler_build_kwargs, sampler_sample_kwargs) = pickle.load(f)
        BathroomTileExperiment(sampler, graph=graph, gamma=gamma, num_reads=num_reads,
                               sampler_build_kwargs=sampler_build_kwargs,
                               sampler_sample_kwargs=sampler_sample_kwargs)

    def run_experiment_or_load(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        data = BathroomTileExperiment.load_if_available(dirpath)
        if not data:
            sampler = self.sampler(**self.sampler_build_kwargs)
            results = sampler.sample_ising(self.graph.hs, self.graph.edges, num_reads=self.num_reads,
                                           **self.sampler_sample_kwargs)
            data = results.data()
            energies = results.energy()
            scalars = results.scalars()

            BathroomTileExperiment.save_data(dirpath, data, energies, scalars=scalars)

    @staticmethod
    def save_data(dirpath, data, energies, scalars=None):
        numpy.save(os.path.join(dirpath, 'data'), data)
        numpy.save(os.path.join(dirpath, 'energy'), energies)
        if scalars:
            pickle.dump(os.path.join(dirpath, 'scalars.pickle'), scalars)

    @staticmethod
    def load_if_available(dirpath):
        if os.path.exists(dirpath):
            datapath = os.path.join(dirpath, 'data.npy')
            data = numpy.load(datapath)

            energypath = os.path.join(dirpath, 'energy.npy')
            energy = numpy.load(energypath)

            scalarpath = os.path.join(dirpath, 'scalars.pickle')
            if os.path.exists(scalarpath):
                with open(scalarpath, 'rb') as f:
                    scalars = pickle.load(f)
            else:
                scalars = {}

            return data, energy, scalars
        else:
            return None
