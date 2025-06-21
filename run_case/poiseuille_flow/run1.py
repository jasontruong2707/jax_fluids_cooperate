import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_animation

# SETUP SIMULATION
input_manager = InputManager("poiseuille.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure"]
jxf_data = load_data(path, quantities)

cell_centers = jxf_data.cell_centers
data = jxf_data.data
times = jxf_data.times

# Prepare plot dict (velocityX assumed along x axis)
plot_dict = {
    "density": data["density"], 
    "velocityX": data["velocity"][:, 0],  # velocity x-component
    "pressure": data["pressure"]
}
x, y, z = cell_centers  # unpack cell centers

# CREATE ANIMATION (unchanged)
create_1D_animation(
    plot_dict,
    cell_centers,
    times,
    nrows_ncols=(1, 3),
    axis="y", axis_values=[0.0, 0.0],
    interval=100
)

# -------------------------
# Analytical Setup
# -------------------------
# Power-law fluid parameters
n = 0.5      # power-law index
k = 0.01        # consistency index
gradP = -0.02   # pressure gradient (Pa/m)
ly = 2.0       # domain height
h = 1.0        # half domain

# Newtonian viscosity (for Vc calculation)
Vc_newtonian = (n / (n + 1.0)) * (-gradP / (k))**(1.0 / n) * h**(1.0 + 1.0 / n)

# Unnormalized analytical power-law velocity
def power_law_velocity(y, gradP, k, n, h):
    term1 = (n) / (n+1)
    term2 = (-gradP / (k)) ** (1.0 / n)
    
    term3 =  (h**(1+1/n) - np.abs(y) ** (1 + 1.0 / n))
    return term1 * term2 * term3

# Analytical velocities
powerlaw_velocity = power_law_velocity(y, gradP, k, n, h)
powerlaw_normalized_vc = powerlaw_velocity / Vc_newtonian
#powerlaw_normalized_max = powerlaw_velocity / np.max(powerlaw_velocity)

# Simulation velocity (final timestep)
sim_velocity = plot_dict["velocityX"][-1, 0, :, 0]
sim_velocity_normalized_vc = sim_velocity / np.max(sim_velocity)
#sim_velocity_normalized_max = sim_velocity / np.max(sim_velocity)

# -------------------------
# Plotting
# -------------------------
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))

ax.plot(y, sim_velocity_normalized_vc, "r.", label="JXF / $V_c$")
ax.plot(y, powerlaw_normalized_vc, "k--", label="Analytical / $V_c$")
# ax.plot(y, sim_velocity, "b", label="JXF ")
# ax.plot(y, powerlaw_velocity, "g--", label="Analytical ")

ax.set_xlabel(r"$y$")
ax.set_ylabel(r"Normalized $u$")
ax.set_title("Velocity Profile Normalizations")
ax.legend(loc="lower right")
ax.set_box_aspect(0.5)

plt.tight_layout()
plt.savefig("poiseuille_comparison_normalization.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


import pandas as pd

# Save only simulation velocity profile at final timestep
df = pd.DataFrame({
    "y": y,                   # y-direction positions (flattened)
    "velocityX_sim": sim_velocity     # simulation velocity in x-direction
})

df.to_csv("poiseuille_sim_velocity.csv", index=False)
print("Saved simulation velocity profile to 'poiseuille_sim_velocity.csv'")