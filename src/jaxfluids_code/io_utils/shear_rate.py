import os
import json
from typing import Dict, Tuple, Union
from functools import partial

import h5py
import jax
import jax.numpy as jnp


from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class ShearRate():
    def __init__(
        self, 
        domain_information: DomainInformation,
        derivative_stencil: SpatialDerivative,
        ) -> None:

        self.domain_information = domain_information
        self.derivative_stencil = derivative_stencil

    def compute_velocity_gradient(
        self,
        velocity: Array
        ) -> Array:
        cell_sizes = self.domain_information.get_device_cell_sizes()
        active_axes_indices = self.domain_information.active_axes_indices
        print("DEBUG: velocity type:", type(velocity))
        print("DEBUG: velocity value:", velocity)
        nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
        shape = velocity[...,nhx,nhy,nhz].shape
        velocity_grad = []
        for i in range(3):
            velocity_grad.append(
                self.derivative_stencil.derivative_xi(velocity, cell_sizes[i], i, i) if i in active_axes_indices else jnp.zeros(shape)
            )
        velocity_grad = jnp.stack(velocity_grad, axis=1)
        return velocity_grad

    # Calculate gamma_dot here:
    def compute_shear_rate(self, velocity: Array) -> Array:
        """
        Computes scalar shear rate γ̇ from symmetric part of ∇u.

        Input gradient shape: (3, 3, Nx, Ny, Nz)
        Output gamma_dot: (Nx, Ny, Nz)
        """
        grad = self.compute_velocity_gradient(velocity)  # shape: (3, 3, Nx, Ny, Nz)

        # Strain-rate tensor: D_ij = 0.5 * (∂u_j/∂x_i + ∂u_i/∂x_j)
        # grad[i, j, x, y, z] = ∂u_j / ∂x_i

        # Transpose to get ∂u_i / ∂x_j: (j, i, x, y, z)
        grad_T = jnp.swapaxes(grad, 0, 1)  # shape: (3, 3, Nx, Ny, Nz)

        # Symmetrize to get D_ij
        D = 0.5 * (grad + grad_T)  # shape: (3, 3, Nx, Ny, Nz)

        # Compute γ̇ = sqrt(2 * D_ij D_ij)
        D_squared = D ** 2
        gamma_dot = jnp.sqrt(2.0 * jnp.sum(D_squared, axis=(0, 1)))  # sum over i,j → (Nx, Ny, Nz)

        print("D shape:", D.shape)                # (3, 3, Nx, Ny, Nz)
        print("gamma_dot shape:", gamma_dot.shape)  # (Nx, Ny, Nz)

        return gamma_dot