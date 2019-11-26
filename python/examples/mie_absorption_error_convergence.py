import meep as mp
from meep.materials import Al
import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as ps
from math import floor

r = 1.0  # radius of sphere

wvl_min = 2*np.pi*r/10
wvl_max = 2*np.pi*r/2

wvl_key = r  # wavelength that will be checked for error convergence

frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
nfrq = 100

# at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
resolutions = np.arange(15,125,10)
f=open("resolutions.txt","w+")
f.write(f'The tested resolutions are: {resolutions}')
f.close()

dpml = 0.5*wvl_max
dair = 0.5*wvl_max

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y),
              mp.Mirror(mp.Z,phase=-1)]

s = 2*(dpml+dair+r)
cell_size = mp.Vector3(s,s,s)

# is_integrated=True necessary for any planewave source extending into PML
sources = [mp.Source(mp.GaussianSource(frq_cen,fwidth=dfrq,is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0,s,s),
                     component=mp.Ez)]

smoothed_relative_errors = []
unsmoothed_relative_errors = []

for res in resolutions:
    sim = mp.Simulation(resolution=floor(res),
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        symmetries=symmetries,
                        Courant=0.4)

    box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r),size=mp.Vector3(0,2*r,2*r)))
    box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r),size=mp.Vector3(0,2*r,2*r)))
    box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r),size=mp.Vector3(2*r,0,2*r)))
    box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r),size=mp.Vector3(2*r,0,2*r)))
    box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r),size=mp.Vector3(2*r,2*r,0)))
    box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r),size=mp.Vector3(2*r,2*r,0)))

    sim.run(until_after_sources=20)

    freqs = mp.get_flux_freqs(box_x1)
    wvls = [1/f for f in freqs]
    closest_index = (np.abs(np.asarray(wvls) - wvl_key)).argmin()

    box_x1_flux0 = mp.get_fluxes(box_x1)

    sim.reset_meep()

    geometry = [mp.Sphere(material=Al,
                          center=mp.Vector3(),
                          radius=r)]

    sim = mp.Simulation(resolution=floor(res),
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        symmetries=symmetries,
                        geometry=geometry,
                        Courant=0.4)

    box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r),size=mp.Vector3(0,2*r,2*r)))
    box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r),size=mp.Vector3(0,2*r,2*r)))
    box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r),size=mp.Vector3(2*r,0,2*r)))
    box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r),size=mp.Vector3(2*r,0,2*r)))
    box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r),size=mp.Vector3(2*r,2*r,0)))
    box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r),size=mp.Vector3(2*r,2*r,0)))

    sim.run(until_after_sources=200)

    box_x1_flux = mp.get_fluxes(box_x1)
    box_x2_flux = mp.get_fluxes(box_x2)
    box_y1_flux = mp.get_fluxes(box_y1)
    box_y2_flux = mp.get_fluxes(box_y2)
    box_z1_flux = mp.get_fluxes(box_z1)
    box_z2_flux = mp.get_fluxes(box_z2)

    abs_flux = np.asarray(box_x1_flux) - np.asarray(box_x2_flux) + np.asarray(box_y1_flux) - np.asarray(box_y2_flux) + np.asarray(box_z1_flux) - np.asarray(box_z2_flux)
    intensity = np.asarray(box_x1_flux0)/(2*r)**2
    abs_cross_section = np.divide(abs_flux, intensity)
    abs_eff_meep_all = abs_cross_section/(np.pi*r ** 2)
    abs_eff_meep = abs_eff_meep_all[closest_index]
    abs_eff_theory = ps.MieQ(np.sqrt(Al.epsilon(freqs[closest_index])[0,0]),1000/freqs[closest_index],2*r*1000,asDict=True)['Qabs']
    relative_error = abs(abs_eff_meep-abs_eff_theory)/abs_eff_theory
    smoothed_relative_errors.append(relative_error)

    sim.reset_meep()

    Al.do_averaging = False

    geometry = [mp.Sphere(material=Al,
                          center=mp.Vector3(),
                          radius=r)]

    sim = mp.Simulation(resolution=floor(res),
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        sources=sources,
                        k_point=mp.Vector3(),
                        symmetries=symmetries,
                        geometry=geometry,
                        Courant=0.4)

    box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r, 2 * r)))
    box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0, 2 * r)))
    box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2 * r, 2 * r, 0)))
    box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2 * r, 2 * r, 0)))

    sim.run(until_after_sources=200)

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
    abs_eff_meep_all = abs_cross_section / (np.pi * r ** 2)
    abs_eff_meep = abs_eff_meep_all[closest_index]
    abs_eff_theory = \
    ps.MieQ(np.sqrt(Al.epsilon(freqs[closest_index])[0, 0]), 1000 / freqs[closest_index], 2 * r * 1000, asDict=True)[
        'Qabs']
    relative_error = abs(abs_eff_meep - abs_eff_theory) / abs_eff_theory
    unsmoothed_relative_errors.append(relative_error)

f = open("smoothed_relative_errors.txt","w+")
f.write(f'The relative errors found with pixel smoothing on are: {smoothed_relative_errors}')
f.close()

f = open("unsmoothed_relative_errors.txt","w+")
f.write(f'The relative errors found without pixel smoothing on are: {unsmoothed_relative_errors}')
f.close()


if mp.am_master():
    plt.figure(dpi=150)
    plt.loglog(resolutions, smoothed_relative_errors, 'bo-', label='With Pixel Smoothing')
    plt.loglog(resolutions, unsmoothed_relative_errors, 'ro-', label='Without Pixel Smoothing')
    plt.grid(True,which="both",ls="-")
    plt.xlabel('resolution (pixels/μm)')
    plt.ylabel('relative error in MEEP calculation (compared to PyMieScatt)')
    plt.legend(loc='upper right')
    plt.title('Relative Error in Mie Absorption Efficiency, Comparing MEEP and PyMieScatt')
    plt.tight_layout()
    plt.savefig("mie_absorption_error.png")
