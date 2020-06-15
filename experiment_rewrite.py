import bathroom_tile
import numpy
import pickle
import os


class BathroomTileExperiment:
    def __init__(self, sampler, unit_cell_rect=None, base_ej_over_kt=39.72, j=1.0, gamma=0.0, graph=None,
                 num_reads=10000, graph_build_kwargs=None, sampler_build_kwargs=None, sampler_sample_kwargs=None):
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
        self.j = j
        self.gamma = gamma
        self.sampler = sampler
        self.sampler_build_kwargs = sampler_build_kwargs or {}
        self.sampler_sample_kwargs = sampler_sample_kwargs or {}
        self.num_reads = num_reads
        self.base_ej_over_kt = base_ej_over_kt

    def save_experiment_config(self, filepath):
        basedir = os.path.dirname(filepath)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        with open(filepath, 'wb') as w:
            obj = {
                'sampler': self.sampler,
                'j': self.j,
                'graph': self.graph,
                'gamma': self.gamma,
                'num_reads': self.num_reads,
                'base_ej_over_kt': self.base_ej_over_kt,
                'sampler_build_kwargs': self.sampler_build_kwargs,
                'sampler_sample_kwargs': self.sampler_sample_kwargs
            }
            pickle.dump(obj, w)

    @staticmethod
    def load_experiment_config(filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return BathroomTileExperiment(obj['sampler'],
                                      j=obj['j'],
                                      graph=obj['graph'],
                                      gamma=obj['gamma'],
                                      num_reads=obj['num_reads'],
                                      base_ej_over_kt=obj['base_ej_over_kt'],
                                      sampler_build_kwargs=obj['sampler_build_kwargs'],
                                      sampler_sample_kwargs=obj['sampler_sample_kwargs'])

    def run_experiment_or_load(self, dirpath, allow_run=True):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        results = BathroomTileExperiment.load_if_available(dirpath)
        if results is None and allow_run:
            sampler = self.sampler(**self.sampler_build_kwargs)
            results = sampler.sample_ising(self.graph.hs, self.graph.edges, transverse_field=self.gamma,
                                           num_reads=self.num_reads, auto_scale=False,
                                           **self.sampler_sample_kwargs)
            data = results.data()
            energies = results.energy()
            scalars = results.scalars() or {}

            # Now add the basic scalars
            scalars.update({
                "ej_by_kt": self.j*self.base_ej_over_kt,
                "j": self.j,
                "gamma": self.gamma
            })

            BathroomTileExperiment.save_data(dirpath, data, energies, scalars=scalars)
            return data, energies, scalars
        else:
            return results

    @staticmethod
    def save_data(dirpath, data, energies, scalars=None):
        numpy.save(os.path.join(dirpath, 'data'), data)
        numpy.save(os.path.join(dirpath, 'energy'), energies)
        if scalars:
            with open(os.path.join(dirpath, 'scalars.pickle'), 'wb') as w:
                pickle.dump(scalars, w)

    @staticmethod
    def load_if_available(dirpath):
        datapath = os.path.join(dirpath, 'data.npy')
        energypath = os.path.join(dirpath, 'energy.npy')
        scalarpath = os.path.join(dirpath, 'scalars.pickle')
        if os.path.exists(datapath) and os.path.exists(energypath):
            data = numpy.load(datapath)
            energy = numpy.load(energypath)

            if os.path.exists(scalarpath):
                with open(scalarpath, 'rb') as f:
                    scalars = pickle.load(f)
            else:
                scalars = {}

            return data, energy, scalars
        else:
            return None
