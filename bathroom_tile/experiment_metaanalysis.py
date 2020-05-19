from matplotlib import pyplot
import scipy.interpolate
import numpy
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


def flippable_phase(base_dir, scalars):
    if 'actual_j' in scalars:
        # This name is weird, its actually the multiplier for j
        ratio = scalars['actual_j']
        ej_over_kt = numpy.asarray(scalars['ej_by_kt']) * ratio
    else:
        ej_over_kt = scalars['ej_by_kt']
    gamma = scalars['gamma']
    flippable_count = scalars['flippable_count']

    kt_over_ej = [1./jk for jk in ej_over_kt]

    # Make a grid
    mgrid_x, mgrid_y = numpy.meshgrid(numpy.linspace(min(gamma), max(gamma), 1000),
                                      numpy.linspace(min(kt_over_ej), max(kt_over_ej), 1000))
    mgrid_z = scipy.interpolate.griddata((gamma, kt_over_ej), flippable_count, (mgrid_x, mgrid_y), method='cubic')

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.savefig(os.path.join(base_dir, 'flippable_phase.svg'))
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
