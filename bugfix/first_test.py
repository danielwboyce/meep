import meep as mp
from matplotlib import pyplot as plt
import numpy as np
import gdspy

cell_size = mp.Vector3(2,2,0)

Si = mp.Medium(index=3.5)

h = 100.0


# Read in the GDS file
lib = gdspy.GdsLibrary(infile='debug.GDS',units='import',unit=1e-6)
main_cell = lib.top_level()[0]
pol_dict = main_cell.get_polygons(by_spec=True)

# Load the ring geometry
polygons = pol_dict[(0, 0)]
geometry = []
for shape in polygons:
    verts = [mp.Vector3(idx[0],idx[1],-50) for idx in shape]
    geometry.append(mp.Prism(
        vertices=verts,height=h,
        axis=mp.Vector3(0,0,1),material=Si))


sim = mp.Simulation(resolution=200,
                    geometry=geometry,
                    cell_size=cell_size,
                    eps_averaging=False)

sim.init_sim()

eps_data = sim.get_array(size=cell_size,center=mp.Vector3(), component=mp.Dielectric)
plt.figure()
plt.imshow(np.rot90(eps_data), interpolation='spline36', cmap='binary')
plt.axis('off')
plt.tight_layout()
plt.savefig('meep.png')
plt.show()