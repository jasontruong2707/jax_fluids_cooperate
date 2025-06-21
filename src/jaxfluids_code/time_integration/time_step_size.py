import jax
from jax import Array
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.numerical_setup import NumericalSetup
#from jaxfluids.stencils.derivative.deriv_center_4 import DerivativeFourthOrderCenter
from jaxfluids.stencils.derivative import DICT_FIRST_DERIVATIVE_CENTER

from jaxfluids.config import precision
from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.levelset.geometry.mask_functions import compute_fluid_masks, compute_cut_cell_mask_sign_change_based

def compute_velocity_gradient_cc_no_slice(velocity: Array, domain_information: DomainInformation, numerical_setup: NumericalSetup) -> Array:
    """Computes the velocity gradient at cell centers.

    :param velocity: Velocity buffer
    :type velocity: Array
    :param domain_information: Domain information
    :type domain_information: DomainInformation
    :return: Velocity gradient tensor
    :rtype: Array
    """
    active_axes_indices = domain_information.active_axes_indices
    cell_sizes = domain_information.get_device_cell_sizes()
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    stencil_type = numerical_setup.conservatives.dissipative_fluxes.derivative_stencil_center  # should be a string like "CENTRAL4"
    StencilClass = stencil_type
    stencil = StencilClass(
        nh=domain_information.nh_conservatives,
        inactive_axes=domain_information.inactive_axes,
        offset=0,
        is_mesh_stretching=domain_information.is_mesh_stretching,
        cell_sizes=cell_sizes,
    )

    velocity_gradient = []
    for axis in range(3):
        if axis in active_axes_indices:
            velocity_gradient.append(stencil.derivative_xi(velocity, cell_sizes[axis], axis))
        else:
            velocity_gradient.append(jnp.zeros_like(velocity[...,nhx,nhy,nhz]))
    velocity_gradient = jnp.stack(velocity_gradient, axis=1)

    return velocity_gradient

def compute_shear_rate(velocity_gradient: Array) -> Array:
    """Computes scalar shear rate γ̇ from symmetric part of ∇u.

    Input gradient shape: (3, 3, Nx, Ny, Nz)
    Output gamma_dot: (Nx, Ny, Nz)
    """
    # Strain-rate tensor: D_ij = 0.5 * (∂u_j/∂x_i + ∂u_i/∂x_j)
    # grad[i, j, x, y, z] = ∂u_j / ∂x_i

    # Transpose to get ∂u_i / ∂x_j: (j, i, x, y, z)
    velocity_gradient_trans = jnp.swapaxes(velocity_gradient, 0, 1)  # shape: (3, 3, Nx, Ny, Nz)

    # Symmetrize to get D_ij
    D = 0.5 * (velocity_gradient + velocity_gradient_trans)  # shape: (3, 3, Nx, Ny, Nz)

    # Compute γ̇ = sqrt(2 * D_ij D_ij)
    D_squared = D ** 2
    gamma_dot = jnp.sqrt(2.0 * jnp.sum(D_squared, axis=(0, 1)))  # sum over i,j → (Nx, Ny, Nz)

    return gamma_dot

def compute_time_step_size(
        primitives: Array,
        temperature: Array,
        levelset: Array,
        volume_fraction: Array,
        solid_temperature: Array,
        domain_information: DomainInformation,
        equation_information: EquationInformation,
        material_manager: MaterialManager,
        solid_properties_manager: SolidPropertiesManager,
        numerical_setup: NumericalSetup
    ) -> float:
    """Computes the physical time step size
    depending on the active physics.

    :param primitives: _description_
    :type primitives: Array
    :param levelset: _description_
    :type levelset: Array
    :param volume_fraction: _description_
    :type volume_fraction: Array
    :return: _description_
    :rtype: jnp.float32
    """

    fixed_time_step_size = numerical_setup.conservatives.time_integration.fixed_timestep

    if fixed_time_step_size:
        dt = fixed_time_step_size

    else:
        EPS = precision.get_eps()

        # DOMAIN INFORMATION
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
        active_axes_indices = domain_information.active_axes_indices
        active_physics_setup = numerical_setup.active_physics

        equation_type = equation_information.equation_type
        levelset_model = equation_information.levelset_model
        solid_coupling = equation_information.solid_coupling

        # First min over cell_sizes in i direction, then min over axes.
        # Necessary for mesh stretching.
        min_cell_size = domain_information.smallest_cell_size
        min_cell_size_squared = min_cell_size * min_cell_size

        ids_mass = equation_information.ids_mass
        ids_energy = equation_information.ids_energy
        ids_volume_fraction = equation_information.ids_volume_fraction
        component_ids = equation_information.ids_species

        s_mass = equation_information.s_mass
        s_energy = equation_information.s_energy
        s_volume_fraction = equation_information.s_volume_fraction
        component_slices = equation_information.s_species

        alpha, mass_fraction = None, None
        density = material_manager.get_density(primitives[...,nhx,nhy,nhz])
        pressure = primitives[ids_energy,...,nhx,nhy,nhz]
        if equation_information.is_compute_temperature:
            temperature = temperature[...,nhx,nhy,nhz]

        if equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpha = primitives[(s_volume_fraction,) + (...,nhx,nhy,nhz)]

        # COMPUTE MASKS
        if levelset_model:
            mask_real = compute_fluid_masks(volume_fraction, levelset_model)
            nh_offset = domain_information.nh_offset
            mask_cut_cells = compute_cut_cell_mask_sign_change_based(levelset, nh_offset)
            mask_real *= (1 - mask_cut_cells)

        # CONVECTIVE CONTRIBUTION
        if equation_information.equation_type == "SINGLE-PHASE" \
            and isinstance(material_manager.material, DICT_MATERIAL["BarotropicCavitationFluid"]):
            speed_of_sound = material_manager.get_speed_of_sound_liquid(pressure, density)

        else:
            speed_of_sound = material_manager.get_speed_of_sound(
                primitives[...,nhx,nhy,nhz],
                pressure,
                density,
                volume_fractions=alpha,
                mass_fractions=mass_fraction
            )

        abs_velocity = 0.0
        for i in domain_information.active_axes_indices:
            id_vel = equation_information.ids_velocity[i]
            abs_velocity += (jnp.abs(primitives[id_vel,...,nhx,nhy,nhz]) + speed_of_sound)
            if levelset_model:
                abs_velocity *= mask_real[..., nhx_,nhy_,nhz_]
        dt = min_cell_size / (jnp.max(abs_velocity) + EPS)

        # VISCOUS CONTRIBUTION
        if active_physics_setup.is_viscous_flux:
            const = 3.0 / 14.0
            
            # Calculate velocity gradient and shear rate for non-Newtonian viscosity
            velocity = primitives[equation_information.s_velocity]
            velocity_gradient = compute_velocity_gradient_cc_no_slice(velocity, domain_information, numerical_setup)
            gamma_dot = compute_shear_rate(velocity_gradient)
            
            kinematic_viscosity = material_manager.get_dynamic_viscosity(
                temperature, primitives[...,nhx,nhy,nhz], gamma_dot,
                mass_fractions=None,
            ) / density
            
            if levelset_model:
                kinematic_viscosity = kinematic_viscosity * mask_real[..., nhx_,nhy_,nhz_]
            dt_viscous = const * min_cell_size_squared / (jnp.max(kinematic_viscosity) + EPS)
            dt = jnp.minimum(dt, dt_viscous)

        # HEAT TRANSFER CONTRIBUTION
        if active_physics_setup.is_heat_flux:
            const = 0.1
            cp = material_manager.get_specific_heat_capacity(
                temperature, primitives[...,nhx,nhy,nhz])
            thermal_diffusivity = material_manager.get_thermal_conductivity(
                temperature, primitives[...,nhx,nhy,nhz],
                mass_fractions=None,
            ) / (density * cp)
            if levelset_model:
                thermal_diffusivity = thermal_diffusivity * mask_real[...,nhx_,nhy_,nhz_]
            dt_thermal = const * min_cell_size_squared / (jnp.max(thermal_diffusivity) + EPS)
            dt = jnp.minimum(dt, dt_thermal)

            if solid_coupling.thermal == "TWO-WAY":
                raise NotImplementedError

        # SURFACE TENSION
        # TODO

        # DIFFUSION SHARPENING CONTRIBUTION
        # if self.numerical_setup.diffuse_interface.diffusion_sharpening.is_diffusion_sharpening:
        #     dt_diffusion_sharpening = \
        #         self.diffuse_interface_handler.compute_diffusion_sharpening_timestep(
        #             primitives)
        #     dt = jnp.minimum(dt, dt_diffusion_sharpening)

        # PARALLEL
        if domain_information.is_parallel:
            dt = jax.lax.pmin(dt, axis_name="i")

        CFL = numerical_setup.conservatives.time_integration.CFL
        dt *= CFL

    return dt