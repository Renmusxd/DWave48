from matplotlib import pyplot
import os

def defect_plot(base_dir, scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    defects = scalars['average_defects']
    defects_stdv = scalars['stdv_defects']
    hs = scalars['h']

    pyplot.errorbar(inv_j, defects, yerr=defects_stdv)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(base_dir, 'defects_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.errorbar(ej_by_kt, defects, yerr=defects_stdv)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(base_dir, 'defects_vs_ej_by_kt.svg'))
    pyplot.clf()

    pyplot.errorbar(hs, defects, yerr=defects_stdv)
    pyplot.xlabel('H')
    pyplot.ylabel('Number of defects')
    pyplot.savefig(os.path.join(base_dir, 'defects_vs_hs.svg'))
    pyplot.clf()


def flippable_plot(base_dir, scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    flippable_count = scalars['flippable_count']
    flippable_stdv = scalars['flippable_stdv']
    hs = scalars['h']

    pyplot.errorbar(inv_j, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(base_dir, 'flippable_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.errorbar(ej_by_kt, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(base_dir, 'flippable_vs_ej_by_kt.svg'))
    pyplot.clf()

    pyplot.errorbar(hs, flippable_count, yerr=flippable_stdv)
    pyplot.xlabel('H')
    pyplot.ylabel('Number of flippable plaquettes')
    pyplot.savefig(os.path.join(base_dir, 'flippable_vs_hs.svg'))
    pyplot.clf()


def unit_cell_divergence_plot(base_dir, scalars):
    inv_j = scalars['inv_j']
    ej_by_kt = scalars['ej_by_kt']
    divergence = scalars['divergence']

    pyplot.plot(inv_j, divergence)
    pyplot.xlabel('1/J')
    pyplot.ylabel('Boundary Divergence')
    pyplot.savefig(os.path.join(base_dir, 'divergence_vs_inv_j.svg'))
    pyplot.clf()

    pyplot.plot(ej_by_kt, divergence)
    pyplot.xlabel('J/kT')
    pyplot.ylabel('Boundary Divergence')
    pyplot.savefig(os.path.join(base_dir, 'divergence_vs_ej_by_kt.svg'))
    pyplot.clf()
