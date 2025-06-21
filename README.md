"# jax_fluids_cooperate" 


This code base on JAX-FLUIDS (https://github.com/tumaer/JAXFLUIDS).

In this repository, I implemented the Power-Law model for non-Newtonian fluids. 

The main adjustments can be found in /src/jaxfluids_code/solvers/source_term_solver.py, where I calculated gamma_dot from velocity gradient at center cells to contribute into the source term contribution on the rhs of NSE. 
For time step calculation, it can be found inside /src/jaxfluids_code/time_integration/compute_time_step_size.py.
