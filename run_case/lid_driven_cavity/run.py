import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# SETUP SIMULATION
input_manager = InputManager("lid_driven_cavity.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["velocity"]
jxf_data = load_data(path, quantities, step=5)

cell_centers = jxf_data.cell_centers
data = jxf_data.data
times = jxf_data.times

# PLOT
nrows_ncols = (1,2)
plot_dict = {
    "velocityX": data["velocity"][:,0],
    "velocityY": data["velocity"][:,1],
}
x,y,z = cell_centers

# CREATE ANIMATION
create_2D_animation(
    plot_dict,
    cell_centers, 
    times, 
    nrows_ncols=nrows_ncols,
    plane="xy", cmap="seismic",
    interval=100)

# CREATE FIGURE
velX = data["velocity"][-1,0,:,:,0]
velY = data["velocity"][-1,1,:,:,0]
vel_abs = np.sqrt(velX**2 + velY**2)

N = len(x)
u_wall = 0.5
reference_data = np.loadtxt("reference_data.txt")

fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(15,4))
ax[0].streamplot(x, y, velX.T, velY.T, color=vel_abs.T, arrowsize=0)
ax[1].plot(reference_data[:,0], reference_data[:,2], linestyle="None", marker=".", label="Reference")
ax[1].plot(y, velX[N//2,:] / u_wall, label="JXF")
ax[2].plot(reference_data[:,1], reference_data[:,3], linestyle="None", marker=".", label="Reference")
ax[2].plot(x, velY[:,N//2] / u_wall, label="JXF")
# create_2D_figure(plot_dict, cell_centers=cell_centers, plane="xy", plane_value=0.0)
ax[1].set_xlabel(r"$y$")
ax[1].set_ylabel(r"$u / u_w$")
ax[2].set_xlabel(r"$x$")
ax[2].set_ylabel(r"$v / u_w$")
ax[2].legend()
for axi in ax:
    axi.set_box_aspect(1.0)
plt.savefig("lid_driven_cavity.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


# Normalize the velocities
velX_norm = velX / u_wall
velY_norm = velY / u_wall
vel_abs_norm = np.sqrt(velX_norm**2 + velY_norm**2)

# Get sizes
Nx = len(x)
Ny = len(y)

# Create meshgrid
X, Y = np.meshgrid(x, y, indexing="ij")  # shape (Nx, Ny)

# === 1. Export u(y) at vertical centerline ===
u_profile = velX_norm[Nx // 2, :]  # shape (Ny,)
uy_data = np.column_stack((y, u_profile))
np.savetxt("u_vs_y_centerline.csv", uy_data, delimiter=",", header="y,u/u_wall", comments="")

# === 2. Export v(x) at horizontal centerline ===
v_profile = velY_norm[:, Ny // 2]  # shape (Nx,)
vx_data = np.column_stack((x, v_profile))
np.savetxt("v_vs_x_centerline.csv", vx_data, delimiter=",", header="x,v/u_wall", comments="")

# === 3. Export full 2D velocity field for streamline/isocontour ===
full_field = np.column_stack((
    X.flatten(),
    Y.flatten(),
    velX_norm.flatten(),
    velY_norm.flatten(),
    vel_abs_norm.flatten()
))
np.savetxt("velocity_field_streamline.csv", full_field,
           delimiter=",", header="x,y,u/u_wall,v/u_wall,velocity_magnitude/u_wall", comments="")

print("Export completed: u-y, v-x profiles and velocity field.")