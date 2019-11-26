import meep as mp
from meep.materials import Al
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import PyMieScatt as ps
import argparse


def main(args):
    r = 1.0  # radius of sphere

    wvl_min = 2 * np.pi * r / 10
    wvl_max = 2 * np.pi * r / 2

    frq_min = 1 / wvl_max
    frq_max = 1 / wvl_min
    frq_cen = 0.5 * (frq_min + frq_max)
    dfrq = frq_max - frq_min
    nfrq = 100

    # at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
    resolution = floor(args.res)

    dpml = 0.5 * wvl_max
    dair = 0.5 * wvl_max

    pml_layers = [mp.PML(thickness=dpml)]

    symmetries = [mp.Mirror(mp.Y),
                  mp.Mirror(mp.Z, phase=-1)]

    s = 2 * (dpml + dair + r)
    cell_size = mp.Vector3(s, s, s)

    Courant = args.courant

    # is_integrated=True necessary for any planewave source extending into PML
    sources = [mp.Source(mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
                         center=mp.Vector3(-0.5 * s + dpml),
                         size=mp.Vector3(0, s, s),
                         component=mp.Ez)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        symmetries=symmetries,
                        Courant=Courant)

    box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2 * r, 2 * r, 0)))
    box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2 * r, 2 * r, 0)))

    until1 = floor(args.until1)
    sim.run(until_after_sources=until1)

    freqs = mp.get_flux_freqs(box_x1)

    box_x1_flux0 = mp.get_fluxes(box_x1)

    sim.reset_meep()

    geometry = [mp.Sphere(material=Al,
                          center=mp.Vector3(),
                          radius=r)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        symmetries=symmetries,
                        geometry=geometry,
                        Courant=Courant)

    box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2 * r, 2 * r, 0)))
    box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2 * r, 2 * r, 0)))

    until2 = floor(args.until2)
    sim.run(until_after_sources=until2)

    box_x1_flux = mp.get_fluxes(box_x1)
    box_x2_flux = mp.get_fluxes(box_x2)
    box_y1_flux = mp.get_fluxes(box_y1)
    box_y2_flux = mp.get_fluxes(box_y2)
    box_z1_flux = mp.get_fluxes(box_z1)
    box_z2_flux = mp.get_fluxes(box_z2)

    abs_flux = np.asarray(box_x1_flux) - np.asarray(box_x2_flux) + np.asarray(box_y1_flux) - np.asarray(
        box_y2_flux) + np.asarray(box_z1_flux) - np.asarray(box_z2_flux)
    intensity = np.asarray(box_x1_flux0) / (2 * r) ** 2
    abs_cross_section = np.divide(abs_flux, intensity)
    abs_eff_meep = abs_cross_section / (np.pi * r ** 2)
    abs_eff_theory = [ps.MieQ(np.sqrt(Al.epsilon(f)[0, 0]), 1000 / f, 2 * r * 1000, asDict=True)['Qabs'] for f in freqs]

    png_prefix = 'res_{}_courant_{}_until1_{}_until2_{}.'.format(resolution, Courant, until1, until2)
    png_suffix = 'mie_absorption_Al.png'
    png_path = png_prefix + png_suffix

    if mp.am_master():
        plt.figure(dpi=150)
        plt.loglog(2 * np.pi * r * np.asarray(freqs), abs_eff_meep, 'bo-', label='Meep')
        plt.loglog(2 * np.pi * r * np.asarray(freqs), abs_eff_theory, 'ro-', label='theory')
        plt.grid(True, which="both", ls="-")
        plt.xlabel('(sphere circumference)/wavelength, 2πr/λ')
        plt.ylabel('absorption efficiency, σ/πr$^{2}$')
        plt.legend(loc='upper right')
        plt.title('Mie Absorption of an Aluminum Sphere')
        plt.tight_layout()
        plt.savefig(png_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, default=25, help='Resolution (default: 25 pixels/um)')
    parser.add_argument('-courant', type=float, default=0.5, help='Courant condition (default: 0.5, cannot be more than 0.5)')
    parser.add_argument('-until1', type=int, default=10, help='First length for until_after_sources in simulation where cell is empty (default: 10')
    parser.add_argument('-until2', type=int, default=100, help='Second length for until_after_sources in simulation where cell has aluminum sphere (default: 100')
    args = parser.parse_args()
    main(args)