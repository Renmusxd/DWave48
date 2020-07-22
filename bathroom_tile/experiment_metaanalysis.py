from matplotlib import pyplot
import scipy.interpolate
import numpy
import os
from matplotlib import cm

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
    gamma_over_j = numpy.asarray(scalars['gamma']) / numpy.asarray(scalars['j'])

    if 'actual_j' in scalars:
        # This name is weird, its actually the multiplier for j
        ratio = scalars['actual_j']
        ej_over_kt = numpy.asarray(scalars['ej_by_kt']) * ratio
        gamma_over_j = gamma_over_j / ratio
    else:
        ej_over_kt = scalars['ej_by_kt']

    flippable_count = scalars['flippable_count']

    kt_over_ej = [1./jk for jk in ej_over_kt]

    # Make a grid
    mgrid_x, mgrid_y = numpy.meshgrid(numpy.linspace(min(gamma_over_j), max(gamma_over_j), 1000),
                                      numpy.linspace(min(kt_over_ej), max(kt_over_ej), 1000))
    mgrid_z = scipy.interpolate.griddata((gamma_over_j, kt_over_ej), flippable_count, (mgrid_x, mgrid_y), method='linear')

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'flippable_phase.svg'))
    pyplot.clf()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=flippable_count, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'flippable_phase_scatter.svg'))
    pyplot.clf()

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'flippable_subset_phase.svg'))
    pyplot.close()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=flippable_count, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'flippable_subset_phase_scatter.svg'))
    pyplot.close()


def orientation_phase(base_dir, scalars):
    gamma_over_j = numpy.asarray(scalars['gamma']) / numpy.asarray(scalars['j'])

    if 'actual_j' in scalars:
        # This name is weird, its actually the multiplier for j
        ratio = numpy.asarray(scalars['actual_j'])
        ej_over_kt = numpy.asarray(scalars['ej_by_kt']) * ratio
        gamma_over_j = gamma_over_j / ratio
    else:
        ej_over_kt = scalars['ej_by_kt']
    abs_orientation_count = scalars['abs_orientation_count']

    kt_over_ej = numpy.asarray([1./jk for jk in ej_over_kt])

    # Make a grid
    mgrid_x, mgrid_y = numpy.meshgrid(numpy.linspace(min(gamma_over_j), max(gamma_over_j), 1000),
                                      numpy.linspace(min(kt_over_ej), max(kt_over_ej), 1000))
    mgrid_z = scipy.interpolate.griddata((gamma_over_j, kt_over_ej), abs_orientation_count, (mgrid_x, mgrid_y), method='linear')

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'abs_orientation_phase.svg'))
    pyplot.close()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=abs_orientation_count, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'abs_orientation_phase_scatter.svg'))
    pyplot.close()

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'abs_orientation_subset_phase.svg'))
    pyplot.close()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=abs_orientation_count, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'abs_orientation_subset_phase_scatter.svg'))
    pyplot.close()


# def psi_phase(base_dir, scalars):
#     gamma_over_j = numpy.asarray(scalars['gamma']) / numpy.asarray(scalars['j'])
#
#     if 'actual_j' in scalars:
#         # This name is weird, its actually the multiplier for j
#         ratio = numpy.asarray(scalars['actual_j'])
#         ej_over_kt = numpy.asarray(scalars['ej_by_kt']) * ratio
#         gamma_over_j = gamma_over_j / ratio
#     else:
#         ej_over_kt = scalars['ej_by_kt']
#     psi = scalars['psi_order_param']
#
#     kt_over_ej = numpy.asarray([1./jk for jk in ej_over_kt])
#
#     # Make a grid
#     mgrid_x, mgrid_y = numpy.meshgrid(numpy.linspace(min(gamma_over_j), max(gamma_over_j), 1000),
#                                       numpy.linspace(min(kt_over_ej), max(kt_over_ej), 1000))
#     mgrid_z = scipy.interpolate.griddata((gamma_over_j, kt_over_ej), psi, (mgrid_x, mgrid_y), method='linear')
#
#     pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
#     pyplot.xlabel(r'$\Gamma / J$')
#     pyplot.ylabel(r'$kT / J$')
#     pyplot.colorbar()
#     pyplot.savefig(os.path.join(base_dir, 'psi_phase.svg'))
#     pyplot.close()
#
#     pyplot.scatter(gamma_over_j, kt_over_ej, c=psi, cmap='jet')
#     pyplot.xlabel(r'$\Gamma / J$')
#     pyplot.ylabel(r'$kT / J$')
#     pyplot.colorbar()
#     pyplot.savefig(os.path.join(base_dir, 'psi_phase_scatter.svg'))
#     pyplot.close()
#
#     pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
#     pyplot.xlabel(r'$\Gamma / J$')
#     pyplot.ylabel(r'$kT / J$')
#     pyplot.xlim((0, 2.0))
#     pyplot.ylim((0, 2.0))
#     pyplot.colorbar()
#     pyplot.savefig(os.path.join(base_dir, 'psi_subset_phase.svg'))
#     pyplot.close()
#
#     pyplot.scatter(gamma_over_j, kt_over_ej, c=psi, cmap='jet')
#     pyplot.xlabel(r'$\Gamma / J$')
#     pyplot.ylabel(r'$kT / J$')
#     pyplot.xlim((0, 2.0))
#     pyplot.ylim((0, 2.0))
#     pyplot.colorbar()
#     pyplot.savefig(os.path.join(base_dir, 'psi_subset_phase_scatter.svg'))
#     pyplot.close()


def psi_phase(base_dir, scalars):
    gamma_over_j = numpy.asarray(scalars['gamma']) / numpy.asarray(scalars['j'])

    if 'actual_j' in scalars:
        # This name is weird, its actually the multiplier for j
        ratio = numpy.asarray(scalars['actual_j'])
        ej_over_kt = numpy.asarray(scalars['ej_by_kt']) * ratio
        gamma_over_j = gamma_over_j / ratio
    else:
        ej_over_kt = scalars['ej_by_kt']
    psi = scalars['complex_pi_pi_average']

    kt_over_ej = numpy.asarray([1./jk for jk in ej_over_kt])

    # Make a grid
    mgrid_x, mgrid_y = numpy.meshgrid(numpy.linspace(min(gamma_over_j), max(gamma_over_j), 1000),
                                      numpy.linspace(min(kt_over_ej), max(kt_over_ej), 1000))
    mgrid_z = scipy.interpolate.griddata((gamma_over_j, kt_over_ej), psi, (mgrid_x, mgrid_y), method='linear')

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'complex_angle_phase.svg'))
    pyplot.close()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=psi, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'complex_angle_scatter.svg'))
    pyplot.close()

    pyplot.contourf(mgrid_x, mgrid_y, mgrid_z)
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'complex_angle_subset_phase.svg'))
    pyplot.close()

    pyplot.scatter(gamma_over_j, kt_over_ej, c=psi, cmap='jet')
    pyplot.xlabel(r'$\Gamma / J$')
    pyplot.ylabel(r'$kT / J$')
    pyplot.xlim((0, 2.0))
    pyplot.ylim((0, 2.0))
    pyplot.colorbar()
    pyplot.savefig(os.path.join(base_dir, 'complex_angle_subset_phase_scatter.svg'))
    pyplot.close()

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
