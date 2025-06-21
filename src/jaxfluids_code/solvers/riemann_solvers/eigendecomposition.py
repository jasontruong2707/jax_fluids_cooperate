from functools import partial
from typing import Tuple, Union

import jax 
from jax import vmap
import jax.numpy as jnp
import numpy as np

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.config import precision

Array = jax.Array

class Eigendecomposition:
    """ The Eigendecomposition class implements functionality for
    eigendecomposition of the Jacobian matrix. Eigendecomposition
    can be done based on primitive or conservative variables. The 
    frozen state can be calculated from arithmetic or Roe averages.

    eigendecomposition_primitives only returns left and right eigenvectors,
    while eigendecomposition_conservatives additionally returns eigenvalues according
    to a user-specified flux-splitting.

    """

    def __init__(
            self, 
            material_manager: MaterialManager, 
            stencil_size: int, 
            domain_information: DomainInformation, 
            equation_information: EquationInformation,
            frozen_state: str, 
            flux_splitting: str = None
        ) -> None:

        self.eps = precision.get_eps()

        self.material_manager = material_manager
        self.equation_information = equation_information
        self.frozen_state = frozen_state 
        self.flux_splitting = flux_splitting

        # DOMAIN INFORMATION
        self.nh = domain_information.nh_conservatives
        self.nh_geometry = domain_information.nh_geometry
        nx, ny, nz = domain_information.global_number_of_cells
        sx, sy, sz = domain_information.split_factors
        nx_device = int(nx / sx)
        ny_device = int(ny / sy)
        nz_device = int(nz / sz)
        number_cells_device = [nx_device, ny_device, nz_device]
        self.nhx, self.nhy, self.nhz = domain_information.domain_slices_conservatives
        self.nhx_geometry, self.nhy_geometry, self.nhz_geometry = domain_information.domain_slices_geometry

        # EQUATION INFORMATION
        self.equation_type = equation_information.equation_type
        self.s_mass = equation_information.s_mass
        self.ids_mass = equation_information.ids_mass
        self.vel_slices = equation_information.s_velocity
        self.vel_ids = equation_information.ids_velocity
        self.s_energy = equation_information.s_energy
        self.ids_energy = equation_information.ids_energy
        self.s_volume_fraction = equation_information.s_volume_fraction
        self.ids_volume_fraction = equation_information.ids_volume_fraction
        self.vel_minors = equation_information.velocity_minor_axes

        self.vel_id_eigs = [
            [self.vel_minors[0][0]  , self.vel_minors[0][1]  ],
            [self.vel_minors[1][0]  , self.vel_minors[1][1]+1],
            [self.vel_minors[2][0]+1, self.vel_minors[2][1]+1],
        ]

        # SLICES TO GET EACH STENCIL
        half_stencil_size = stencil_size//2
        stencil_index = np.arange(stencil_size).reshape(1,-1) # SLICE
        stencil_indices_i = []
        for i in range(3):
            domain_index_i = np.arange(
                self.nh - half_stencil_size, 
                number_cells_device[i] + self.nh - half_stencil_size + 1
                ).reshape(-1,1)
            slice_i = stencil_index + domain_index_i
            stencil_indices_i.append(slice_i)
        # stencil_indices_i = [
        #     stencil_indices + np.arange(self.nh - half_stencil_size, nx_device + self.nh - half_stencil_size + 1).reshape(-1,1),
        #     stencil_indices + np.arange(self.nh - half_stencil_size, ny_device + self.nh - half_stencil_size + 1).reshape(-1,1),
        #     stencil_indices + np.arange(self.nh - half_stencil_size, nz_device + self.nh - half_stencil_size + 1).reshape(-1,1),
        # ]
        self.stencil_slices = [
            jnp.s_[..., stencil_indices_i[0], self.nhy, self.nhz],
            jnp.s_[..., self.nhx, stencil_indices_i[1], self.nhz],
            jnp.s_[..., self.nhx, self.nhy, stencil_indices_i[2]],
        ]

        # SLICES TO GET CELL I AND I+1
        self.slice_LR = [
            [jnp.s_[..., self.nh-1:-self.nh, self.nhy, self.nhz], jnp.s_[..., self.nh:-self.nh+1, self.nhy, self.nhz]],
            [jnp.s_[..., self.nhx, self.nh-1:-self.nh, self.nhz], jnp.s_[..., self.nhx, self.nh:-self.nh+1, self.nhz]],
            [jnp.s_[..., self.nhx, self.nhy, self.nh-1:-self.nh], jnp.s_[..., self.nhx, self.nhy, self.nh:-self.nh+1]],
        ]

        self.slice_domain_plus_one = [
            jnp.s_[..., self.nh-1:-self.nh+1, self.nhy, self.nhz],
            jnp.s_[..., self.nhx, self.nh-1:-self.nh+1, self.nhz],
            jnp.s_[..., self.nhx, self.nhy, self.nh-1:-self.nh+1],
        ]

        if self.nh_geometry is not None:
            self.slice_geometry = [
                [jnp.s_[..., self.nh_geometry-1:-self.nh_geometry, self.nhy_geometry, self.nhz_geometry], jnp.s_[..., self.nh_geometry:-self.nh_geometry+1, self.nhy_geometry, self.nhz_geometry]],
                [jnp.s_[..., self.nhx_geometry, self.nh_geometry-1:-self.nh_geometry, self.nhz_geometry], jnp.s_[..., self.nhx_geometry, self.nh_geometry:-self.nh_geometry+1, self.nhz_geometry]],
                [jnp.s_[..., self.nhx_geometry, self.nhy_geometry, self.nh_geometry-1:-self.nh_geometry], jnp.s_[..., self.nhx_geometry, self.nhy_geometry, self.nh_geometry:-self.nh_geometry+1]],
            ]

    def get_stencil_window(self, var: Array, axis: int) -> Array:
        return var[self.stencil_slices[axis]]

    def compute_frozen_state(self, 
            primitives: Array,
            curvature: Array,
            axis: int
            ) -> Tuple[Array]:
        """Computes the frozen state at each cell-face in axis direction via
        1) arithmetic mean
        2) Roe average

        :param primitives: _description_
        :type primitives: Array
        :param curvature: _description_
        :type curvature: Array
        :param axis: _description_
        :type axis: int
        :return: _description_
        :rtype: Array
        """

        primitives_L = primitives[self.slice_LR[axis][0]]
        primitives_R = primitives[self.slice_LR[axis][1]]

        if curvature is not None:
            sigma_curvature_ave = self.material_manager.get_sigma() * 0.5 * (curvature[self.slice_geometry[axis][0]] + curvature[self.slice_geometry[axis][1]])
        else:
            sigma_curvature_ave = None
        
        if self.frozen_state == "ARITHMETIC":
            primes_ave = 0.5 * ( primitives_L + primitives_R )
            
            if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
                grueneisen_L = self.material_manager.get_grueneisen(
                    alpha_rho_i=primitives_L[self.s_mass], 
                    alpha_i=primitives_L[self.s_volume_fraction]) 
                grueneisen_R = self.material_manager.get_grueneisen(
                    alpha_rho_i=primitives_R[self.s_mass], 
                    alpha_i=primitives_R[self.s_volume_fraction]) 
                grueneisen_ave = 0.5 * (grueneisen_L + grueneisen_R)

                enthalpy_L = self.material_manager.get_total_enthalpy(
                    primitives_L[self.ids_energy], 
                    primitives_L[self.vel_slices],
                    alpha_rho_i=primitives_L[self.s_mass], 
                    alpha_i=primitives_L[self.s_volume_fraction])
                enthalpy_R = self.material_manager.get_total_enthalpy(
                    primitives_R[self.ids_energy], 
                    primitives_R[self.vel_slices],
                    alpha_rho_i=primitives_R[self.s_mass], 
                    alpha_i=primitives_R[self.s_volume_fraction])
                enthalpy_ave = 0.5 * (enthalpy_L + enthalpy_R)
                
                c_L = self.material_manager.get_speed_of_sound(primitives_L)
                c_R = self.material_manager.get_speed_of_sound(primitives_R)
                c_ave = 0.5 * (c_L + c_R)
                # c_ave = jnp.minimum(c_L, c_R)
                # c_ave = jnp.maximum(c_L, c_R)
                # c_ave = self.material_manager.get_speed_of_sound(primes_ave)

                # grueneisen_ave = self.material_manager.get_grueneisen(
                #     alpha_rho_i=primes_ave[self.s_mass], 
                #     alpha_i=primes_ave[self.s_volume_fraction]) 
                # enthalpy_ave = self.material_manager.get_total_enthalpy(
                #     primes_ave[self.ids_energy], 
                #     primes_ave[self.vel_slices],
                #     alpha_rho_i=primes_ave[self.s_mass], 
                #     alpha_i=primes_ave[self.s_volume_fraction])
                # c_ave = self.material_manager.get_speed_of_sound(primes_ave)
            
            elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
                # grueneisen_L = self.material_manager.get_grueneisen(
                #     alpha_rho_i=primitives_L[self.s_mass], 
                #     alpha_i=primitives_L[self.s_volume_fraction]) 
                # grueneisen_R = self.material_manager.get_grueneisen(
                #     alpha_rho_i=primitives_R[self.s_mass], 
                #     alpha_i=primitives_R[self.s_volume_fraction]) 
                # grueneisen_ave = 0.5 * (grueneisen_L + grueneisen_R)
                grueneisen_ave = None

                # enthalpy_L = self.material_manager.get_total_enthalpy(
                #     primitives_L[self.ids_energy], 
                #     primitives_L[self.vel_slices], 
                #     alpha_rho_i=primitives_L[self.s_mass], 
                #     alpha_i=primitives_L[self.s_volume_fraction])
                # enthalpy_R = self.material_manager.get_total_enthalpy(
                #     primitives_R[self.ids_energy], 
                #     primitives_R[self.vel_slices], 
                #     alpha_rho_i=primitives_R[self.s_mass], 
                #     alpha_i=primitives_R[self.s_volume_fraction])
                # enthalpy_ave = 0.5 * (enthalpy_L + enthalpy_R)
                enthalpy_ave = None
                
                c_L = self.material_manager.get_speed_of_sound(primitives_L)
                c_R = self.material_manager.get_speed_of_sound(primitives_R)
                c_ave = 0.5 * (c_L + c_R)

            elif self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
                temperature_ave = self.material_manager.get_temperature(primes_ave)
                grueneisen_ave = self.material_manager.get_grueneisen(rho=primes_ave[self.ids_mass], T=temperature_ave) \
                    * jnp.ones_like(primes_ave[self.ids_mass])
                enthalpy_ave = self.material_manager.get_total_enthalpy(
                    p=primes_ave[self.ids_energy],
                    velocity_vec=primes_ave[self.vel_slices],
                    rho=primes_ave[self.ids_mass])
                c_ave = self.material_manager.get_speed_of_sound(
                    pressure=primes_ave[self.ids_energy], 
                    density=primes_ave[self.ids_mass])
            
            else:
                raise NotImplementedError
            
            cc_ave = c_ave * c_ave
            velocity_square = jnp.sum(jnp.square(primes_ave[self.vel_slices]), axis=0)
        
        if self.frozen_state == "ROE":
            if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
                # TODO Better way to calculate primes_ave
                primes_ave = self.compute_roe_cons(primitives_L, primitives_R)
                primes_ave = primes_ave.at[0].set(jnp.sqrt(primitives_L[0] * primitives_R[0]))

                rho_sqrt_L, rho_sqrt_R = jnp.sqrt(primitives_L[0]), jnp.sqrt(primitives_R[0])
                rho_div = 1.0 / ( rho_sqrt_L + rho_sqrt_R )
                
                enthalpy_L = self.material_manager.get_total_enthalpy(
                    p=primitives_L[self.ids_energy],
                    velocity_vec=primitives_L[self.vel_slices],
                    rho=primitives_L[self.ids_mass])
                enthalpy_R = self.material_manager.get_total_enthalpy(
                    p=primitives_R[self.ids_energy],
                    velocity_vec=primitives_R[self.vel_slices],
                    rho=primitives_R[self.ids_mass])
                enthalpy_ave = (rho_sqrt_L * enthalpy_L + rho_sqrt_R * enthalpy_R) * rho_div
                
                psi_L = self.material_manager.get_psi(
                    p=primitives_L[self.ids_energy], 
                    rho=primitives_L[self.ids_mass])
                psi_R = self.material_manager.get_psi(
                    p=primitives_R[self.ids_energy], 
                    rho=primitives_R[self.ids_mass])
                psi_ave = (rho_sqrt_L * psi_L + rho_sqrt_R * psi_R) * rho_div

                grueneisen_L = self.material_manager.get_grueneisen(rho=primitives_L[self.ids_mass])
                grueneisen_R = self.material_manager.get_grueneisen(rho=primitives_R[self.ids_mass])
                grueneisen_ave = (rho_sqrt_L * grueneisen_L + rho_sqrt_R * grueneisen_R) * rho_div

                # TODO consistent summation
                squared_velocity_difference = jnp.sum(jnp.square(primitives_R[self.vel_slices] - primitives_L[self.vel_slices]), axis=0)

                p_over_rho_ave = (rho_sqrt_L * primitives_L[4]/primitives_L[0] + rho_sqrt_R * primitives_R[4]/primitives_R[0]) * rho_div \
                                + 0.5 * primes_ave[0] * rho_div * rho_div * squared_velocity_difference

                # TODO consistent summation
                velocity_square = jnp.sum(jnp.square(primes_ave[self.vel_slices]), axis=0)
                
                # cc_ave = (self.material_manager.get_gamma() - 1) * (enthalpy_ave - 0.5 * velocity_square)
                cc_ave = psi_ave + grueneisen_ave * p_over_rho_ave
                c_ave  = jnp.sqrt( cc_ave )
            
            else:
                raise NotImplementedError

        return primes_ave, enthalpy_ave, grueneisen_ave, \
            c_ave, cc_ave, velocity_square, sigma_curvature_ave

    def compute_roe_cons(self, prime_L: Array, prime_R: Array) -> Array:
        """Computes the Roe averaged conservative state.

        :param prime_L: Buffer of primitive variables left of a cell face.
        :type prime_L: Array
        :param prime_R: Buffer of primitive variables right of a cell face.
        :type prime_R: Array
        :return: Buffer of Roe averaged quantities at the cell face.
        :rtype: Array
        """
        roe_cons = (jnp.sqrt(prime_L[0]) * prime_L + jnp.sqrt(prime_R[0]) * prime_R) / (jnp.sqrt(prime_L[0]) + jnp.sqrt(prime_R[0]))
        return roe_cons

    def eigendecomposition_primitives(
            self,
            primitives: Array,
            curvature: Array,
            axis: int
            ) -> Tuple[Array, Array]:
        """Computes the eigendecomposition of the Jacobian matrix wrt primitive variables.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param axis: Direction of the cell face at which the eigendecomposition is to be performed.
        :type axis: int
        :return: Buffer of left and right eigenvectors.
        :rtype: Tuple[Array, Array]
        """

        primes_ave, _, _, c_ave, cc_ave, \
        _, sigma_curvature_ave = self.compute_frozen_state(primitives, curvature, axis)
        _s = primes_ave[0].shape

        right_eigs = [ [jnp.zeros(_s) for _ in range(self.equation_information.no_primes)]
            for _ in range(self.equation_information.no_primes) ]
        left_eigs  = [ [jnp.zeros(_s) for _ in range(self.equation_information.no_primes)]
            for _ in range(self.equation_information.no_primes) ]

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            # RIGHT EIGENVECTORS
            # MASS
            right_eigs[0][0]  = primes_ave[0]
            right_eigs[0][1]  = jnp.ones(_s)
            right_eigs[0][-1] = primes_ave[0]
            # MOMENTA
            right_eigs[self.vel_ids[axis]][0]  = -c_ave
            right_eigs[self.vel_ids[axis]][-1] = c_ave
            right_eigs[self.vel_minors[axis][0]][self.vel_id_eigs[axis][0]] = jnp.ones(_s)
            right_eigs[self.vel_minors[axis][1]][self.vel_id_eigs[axis][1]] = jnp.ones(_s)
            # ENERGY
            right_eigs[4][0]  = cc_ave * primes_ave[0]
            right_eigs[4][-1] = cc_ave * primes_ave[0]

            # LEFT EIGENVECTORS
            # MASS
            left_eigs[1][0]  = jnp.ones(_s)
            # MOMENTA
            left_eigs[0][self.vel_ids[axis]]  = -0.5 / c_ave
            left_eigs[-1][self.vel_ids[axis]] = 0.5 / c_ave
            left_eigs[self.vel_id_eigs[axis][0]][self.vel_minors[axis][0]] = jnp.ones(_s)
            left_eigs[self.vel_id_eigs[axis][1]][self.vel_minors[axis][1]] = jnp.ones(_s)
            # ENERGY
            left_eigs[0][4]  = 0.5 / (cc_ave * primes_ave[0])
            left_eigs[1][4]  = -1.0 / cc_ave
            left_eigs[-1][4] = 0.5 / (cc_ave * primes_ave[0])

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho_ave = self.material_manager.get_density(primes_ave)

            # RIGHT EIGENVECTORS
            for mass_id in self.ids_mass:
                right_eigs[mass_id][0]            = primes_ave[mass_id]
                right_eigs[mass_id][mass_id + 1]  = jnp.ones(_s)
                right_eigs[mass_id][-1]           = primes_ave[mass_id]

            right_eigs[self.vel_ids[axis]][0]  = -c_ave
            right_eigs[self.vel_ids[axis]][-1] = c_ave
            right_eigs[self.vel_minors[axis][0]][self.vel_id_eigs[axis][0]] = jnp.ones(_s)
            right_eigs[self.vel_minors[axis][1]][self.vel_id_eigs[axis][1]] = jnp.ones(_s)
            right_eigs[self.ids_energy][0]  = cc_ave * rho_ave
            if sigma_curvature_ave is not None:
                right_eigs[self.ids_energy][-2] = sigma_curvature_ave
            right_eigs[self.ids_energy][-1] = cc_ave * rho_ave

            for vf_id in self.ids_volume_fraction:
                right_eigs[vf_id][vf_id-1] = jnp.ones(_s)

            # LEFT EIGENVECTORS
            one_c_ave = 1.0 / c_ave
            one_rho_cc_ave = 1.0 / (cc_ave * rho_ave)

            for mass_id in self.ids_mass:
                left_eigs[mass_id+1][mass_id] = jnp.ones(_s)

            left_eigs[0][self.vel_ids[axis]]  = -0.5 * one_c_ave
            left_eigs[-1][self.vel_ids[axis]] = 0.5 * one_c_ave
            left_eigs[self.vel_id_eigs[axis][0]][self.vel_minors[axis][0]] = jnp.ones(_s)
            left_eigs[self.vel_id_eigs[axis][1]][self.vel_minors[axis][1]] = jnp.ones(_s)
            left_eigs[0][self.ids_energy]  = 0.5 * one_rho_cc_ave
            left_eigs[-1][self.ids_energy] = 0.5 * one_rho_cc_ave
            if sigma_curvature_ave is not None:
                left_eigs[0][-1] = - 0.5 * sigma_curvature_ave * one_rho_cc_ave
                left_eigs[-1][-1] = - 0.5 * sigma_curvature_ave * one_rho_cc_ave

            for mass_id in self.ids_mass:
                left_eigs[mass_id+1][self.ids_energy] = -primes_ave[mass_id] * one_rho_cc_ave
                if sigma_curvature_ave is not None:
                    left_eigs[mass_id+1][-1] = primes_ave[mass_id] * sigma_curvature_ave * one_rho_cc_ave

            for vf_id in self.ids_volume_fraction:
                left_eigs[vf_id-1][vf_id] = jnp.ones(_s)
        
        else:
            raise NotImplementedError

        right_eigs = jnp.array(right_eigs)
        left_eigs = jnp.array(left_eigs)
        return right_eigs, left_eigs

    def get_characteristics_from_primitives(
            self,
            xin_list: Tuple[Array],
            primitives: Array,
            curvature: Array,
            axis: int
            ) -> Tuple[Array, Array]:
        
        primes_ave, _, _, c_ave, cc_ave, \
        _, sigma_curvature_ave = self.compute_frozen_state(primitives, curvature, axis)
        
        primes_ave = jnp.expand_dims(primes_ave, axis=axis-3)
        c_ave = jnp.expand_dims(c_ave, axis=axis-3)
        cc_ave = jnp.expand_dims(cc_ave, axis=axis-3)
        if sigma_curvature_ave is not None:
            sigma_curvature_ave = jnp.expand_dims(sigma_curvature_ave, axis=axis-3)
        
        xout_list = []

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]
                
                xout[self.ids_mass] = -0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 / (cc_ave * primes_ave[self.ids_mass]) * xin[self.ids_energy]
                xout[1] = xin[self.ids_mass] - 1.0 / cc_ave * xin[self.ids_energy]
                xout[self.vel_id_eigs[axis][0]] = xin[self.vel_minors[axis][0]]
                xout[self.vel_id_eigs[axis][1]] = xin[self.vel_minors[axis][1]]
                xout[self.ids_energy] = 0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 / (cc_ave * primes_ave[self.ids_mass]) * xin[self.ids_energy]
                
                xout_list.append(jnp.array(xout))

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho_ave = self.material_manager.get_density(primes_ave)
            one_cc_rho_ave = 1.0 / (cc_ave * rho_ave)

            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]

                for mass_id in self.ids_mass:
                    xout[mass_id+1] = xin[mass_id] \
                        - one_cc_rho_ave * primes_ave[mass_id] * xin[self.ids_energy]
                    if sigma_curvature_ave is not None:
                        xout[mass_id+1] += one_cc_rho_ave * primes_ave[mass_id] * sigma_curvature_ave * xin[-1]

                xout[0] = -0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 * one_cc_rho_ave * xin[self.ids_energy]
                xout[-1] = 0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 * one_cc_rho_ave * xin[self.ids_energy]

                if sigma_curvature_ave is not None:
                    xout[0] += - 0.5 * one_cc_rho_ave * sigma_curvature_ave * xin[-1]
                    xout[-1] += - 0.5 * one_cc_rho_ave * sigma_curvature_ave * xin[-1]
                
                xout[self.vel_id_eigs[axis][0]] = xin[self.vel_minors[axis][0]]
                xout[self.vel_id_eigs[axis][1]] = xin[self.vel_minors[axis][1]]

                for vf_id in self.ids_volume_fraction:
                    xout[vf_id-1] = xin[vf_id]

                xout_list.append(jnp.array(xout))
        
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            rho_ave = self.material_manager.get_density(primes_ave)
            one_cc_rho_ave = 1.0 / (cc_ave * rho_ave)

            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]

                for mass_id in self.ids_mass:
                    xout[mass_id+1] = xin[mass_id] \
                        - one_cc_rho_ave * primes_ave[mass_id] * xin[self.ids_energy]
                    if sigma_curvature_ave is not None:
                        raise NotImplementedError
                        # xout[mass_id+1] += one_cc_rho_ave * primes_ave[mass_id] * sigma_curvature_ave * xin[-1]

                xout[0] = -0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 * one_cc_rho_ave * xin[self.ids_energy]
                xout[-1] = 0.5 / c_ave * xin[self.vel_ids[axis]] \
                    + 0.5 * one_cc_rho_ave * xin[self.ids_energy]

                if sigma_curvature_ave is not None:
                    raise NotImplementedError
                    # xout[0] += - 0.5 * one_cc_rho_ave * sigma_curvature_ave * xin[-1]
                    # xout[-1] += - 0.5 * one_cc_rho_ave * sigma_curvature_ave * xin[-1]
                
                xout[self.vel_id_eigs[axis][0]] = xin[self.vel_minors[axis][0]]
                xout[self.vel_id_eigs[axis][1]] = xin[self.vel_minors[axis][1]]

                xout_list.append(jnp.array(xout))
        
        else:
            raise NotImplementedError

        return xout_list
    
    def get_primitives_from_characteristics(
            self,
            xin_list: Tuple[Array],
            primitives: Array,
            curvature: Array,
            axis: int
            ) -> Tuple[Array, Array]:
        
        primes_ave, _, _, c_ave, cc_ave, \
        _, sigma_curvature_ave = self.compute_frozen_state(primitives, curvature, axis)
        
        xout_list = []

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            
            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]
                
                xout[self.ids_mass] = primes_ave[self.ids_mass] * (xin[0] + xin[4]) + xin[1]
                xout[self.vel_ids[axis]] = c_ave * (-xin[0] + xin[4])
                xout[self.vel_minors[axis][0]] = xin[self.vel_id_eigs[axis][0]]
                xout[self.vel_minors[axis][1]] = xin[self.vel_id_eigs[axis][1]]
                xout[self.ids_energy] = cc_ave * primes_ave[self.ids_mass] * (xin[0] + xin[4])
                
                xout_list.append(jnp.array(xout))

        elif self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho_ave = self.material_manager.get_density(primes_ave)

            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]

                for mass_id in self.ids_mass:
                    xout[mass_id] = primes_ave[mass_id] * (xin[0] + xin[-1]) \
                        + xin[mass_id + 1]

                xout[self.vel_ids[axis]] = c_ave * (-xin[0] + xin[-1])
                xout[self.vel_minors[axis][0]] = xin[self.vel_id_eigs[axis][0]]
                xout[self.vel_minors[axis][1]] = xin[self.vel_id_eigs[axis][1]]

                xout[self.ids_energy] = cc_ave * rho_ave * (xin[0] + xin[-1])
                if sigma_curvature_ave is not None:
                    # TODO check index -2 for more than 2 components
                    xout[self.ids_energy] += sigma_curvature_ave * xin[-2]

                for vf_id in self.ids_volume_fraction:
                    xout[vf_id] = xin[vf_id-1]
            
                xout_list.append(jnp.array(xout))
        
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            rho_ave = self.material_manager.get_density(primes_ave)

            for xin in xin_list:
                xout = [0 for _ in range(self.equation_information.no_primes)]

                for mass_id in self.ids_mass:
                    xout[mass_id] = primes_ave[mass_id] * (xin[0] + xin[-1]) \
                        + xin[mass_id + 1]

                xout[self.vel_ids[axis]] = c_ave * (-xin[0] + xin[-1])
                xout[self.vel_minors[axis][0]] = xin[self.vel_id_eigs[axis][0]]
                xout[self.vel_minors[axis][1]] = xin[self.vel_id_eigs[axis][1]]

                xout[self.ids_energy] = cc_ave * rho_ave * (xin[0] + xin[-1])
                if sigma_curvature_ave is not None:
                    raise NotImplementedError
                    # TODO check index -2 for more than 2 components
                    # xout[self.ids_energy] += sigma_curvature_ave * xin[-2]
            
                xout_list.append(jnp.array(xout))
        
        else:
            raise NotImplementedError

        return xout_list
   
    def eigendecomposition_conservatives(
            self,
            primitives: Array,
            axis: int
            ) -> Union[Tuple[Array, Array],
        Tuple[Array, Array, Array]]:
        """Computes eigendecomposition of the Jacobian matrix for conservative variables.
        Formulation for a general equation of state. Implementation according to Fedkiv et al.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param axis: Direction of the cell face at which the eigendecomposition is to be performed.
        :type axis: int
        :return: Buffer of left, right eigenvectors and the eigenvalues.
        :rtype: Union[Tuple[Array, Array], Tuple[Array, Array, Array]]
        """

        primes_ave, enthalpy_ave, grueneisen_ave, c_ave, cc_ave, \
        velocity_square, sigma_curvature_ave = self.compute_frozen_state(primitives, None, axis)
        one_cc_ave  = 1.0 / cc_ave
        one_rho_ave = 1.0 / primes_ave[0]
        _s = primes_ave[0].shape

        # LEFT AND RIGHT EIGENVECTORS ACCORDING TO FEDKIW ET AL 1999
        right_eigs = [ [jnp.zeros(_s) for ii in range(self.equation_information.no_primes)]
            for ii in range(self.equation_information.no_primes) ]
        left_eigs  = [ [jnp.zeros(_s) for ii in range(self.equation_information.no_primes)] 
            for ii in range(self.equation_information.no_primes) ]

        if self.equation_information.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            # RIGHT EIGENVECTORS
            # MASS
            right_eigs[0][0]                        = jnp.ones(_s)
            right_eigs[self.vel_ids[axis]][0]       = primes_ave[self.vel_ids[axis]] - c_ave
            right_eigs[self.vel_minors[axis][0]][0] = primes_ave[self.vel_minors[axis][0]] 
            right_eigs[self.vel_minors[axis][1]][0] = primes_ave[self.vel_minors[axis][1]] 
            right_eigs[4][0]                        = enthalpy_ave - primes_ave[self.vel_ids[axis]] * c_ave 
            # MOMENTA
            # MAJOR
            right_eigs[0][self.vel_ids[axis]] = grueneisen_ave
            right_eigs[1][self.vel_ids[axis]] = grueneisen_ave * primes_ave[1]
            right_eigs[2][self.vel_ids[axis]] = grueneisen_ave * primes_ave[2]
            right_eigs[3][self.vel_ids[axis]] = grueneisen_ave * primes_ave[3]
            right_eigs[4][self.vel_ids[axis]] = grueneisen_ave * enthalpy_ave - cc_ave
            # MINOR - 1
            right_eigs[self.vel_minors[axis][0]][self.vel_minors[axis][0]] = -primes_ave[0]
            right_eigs[4][self.vel_minors[axis][0]]                        = -primes_ave[0] * primes_ave[self.vel_minors[axis][0]]
            # MINOR - 2
            right_eigs[self.vel_minors[axis][1]][self.vel_minors[axis][1]] = primes_ave[0]
            right_eigs[4][self.vel_minors[axis][1]]                        = primes_ave[0] * primes_ave[self.vel_minors[axis][1]]
            # ENERGY
            right_eigs[0][4]                        = jnp.ones(_s)
            right_eigs[self.vel_ids[axis]][4]       = primes_ave[self.vel_ids[axis]] + c_ave
            right_eigs[self.vel_minors[axis][0]][4] = primes_ave[self.vel_minors[axis][0]] 
            right_eigs[self.vel_minors[axis][1]][4] = primes_ave[self.vel_minors[axis][1]] 
            right_eigs[4][4]                        = enthalpy_ave + primes_ave[self.vel_ids[axis]] * c_ave 

            # LEFT EIGENVECTORS
            # MASS
            left_eigs[0][0]                        = 0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave + (primes_ave[self.vel_ids[axis]] + c_ave) * c_ave )
            left_eigs[0][self.vel_ids[axis]]       = 0.5 * one_cc_ave * (-primes_ave[self.vel_ids[axis]] * grueneisen_ave - c_ave)  
            left_eigs[0][self.vel_minors[axis][0]] = 0.5 * one_cc_ave * (-primes_ave[self.vel_minors[axis][0]] * grueneisen_ave)
            left_eigs[0][self.vel_minors[axis][1]] = 0.5 * one_cc_ave * (-primes_ave[self.vel_minors[axis][1]] * grueneisen_ave)
            left_eigs[0][4]                        = 0.5 * one_cc_ave * grueneisen_ave
            # MOMENTA
            # MAJOR
            left_eigs[self.vel_ids[axis]][0] = one_cc_ave * (enthalpy_ave - velocity_square) 
            left_eigs[self.vel_ids[axis]][1] = primes_ave[1] * one_cc_ave
            left_eigs[self.vel_ids[axis]][2] = primes_ave[2] * one_cc_ave
            left_eigs[self.vel_ids[axis]][3] = primes_ave[3] * one_cc_ave
            left_eigs[self.vel_ids[axis]][4] = -one_cc_ave
            # MINOR - 1
            left_eigs[self.vel_minors[axis][0]][0]                        = primes_ave[self.vel_minors[axis][0]] * one_rho_ave
            left_eigs[self.vel_minors[axis][0]][self.vel_minors[axis][0]] = -one_rho_ave
            # MINOR - 2
            left_eigs[self.vel_minors[axis][1]][0]                        = -primes_ave[self.vel_minors[axis][1]] * one_rho_ave
            left_eigs[self.vel_minors[axis][1]][self.vel_minors[axis][1]] = one_rho_ave
            # ENERGY
            left_eigs[4][0]                        = 0.5 * one_cc_ave * (grueneisen_ave * velocity_square - grueneisen_ave * enthalpy_ave - (primes_ave[self.vel_ids[axis]] - c_ave) * c_ave )
            left_eigs[4][self.vel_ids[axis]]       = 0.5 * one_cc_ave * (-primes_ave[self.vel_ids[axis]] * grueneisen_ave + c_ave)  
            left_eigs[4][self.vel_minors[axis][0]] = 0.5 * one_cc_ave * (-primes_ave[self.vel_minors[axis][0]] * grueneisen_ave)
            left_eigs[4][self.vel_minors[axis][1]] = 0.5 * one_cc_ave * (-primes_ave[self.vel_minors[axis][1]] * grueneisen_ave)
            left_eigs[4][4]                        = 0.5 * one_cc_ave * grueneisen_ave 

        elif self.equation_information.equation_type == "DIFFUSE-INTERFACE-5EQM":
            raise NotImplementedError

        else:
            raise NotImplementedError

        right_eigs, left_eigs = jnp.array(right_eigs), jnp.array(left_eigs)

        if not self.flux_splitting:
            return right_eigs, left_eigs

        # EIGENVALUES FOR FLUX-SPLITTING

        # ROE EIGENVALUES
        if self.flux_splitting == "ROE":
            gamma_1   = jnp.abs(primes_ave[axis+1] - c_ave)
            gamma_234 = jnp.abs(primes_ave[axis+1])
            gamma_5   = jnp.abs(primes_ave[axis+1] + c_ave) 

        # cLLF EIGENVALUES
        elif self.flux_splitting == "CLLF":
            primitives_L = primitives[self.slice_LR[axis][0]]
            primitives_R = primitives[self.slice_LR[axis][1]]
            c_L = self.material_manager.get_speed_of_sound(primitives_L)
            c_R = self.material_manager.get_speed_of_sound(primitives_R)
            gamma_1   = jnp.maximum(jnp.abs(primitives_L[axis+1] - c_L), jnp.abs(primitives_R[axis+1] - c_R))
            gamma_234 = jnp.maximum(jnp.abs(primitives_L[axis+1]), jnp.abs(primitives_R[axis+1]))
            gamma_5   = jnp.maximum(jnp.abs(primitives_L[axis+1] + c_L), jnp.abs(primitives_R[axis+1] + c_R))

        # LLF EIGENVALUES
        elif self.flux_splitting == "LLF":
            primitives_L = primitives[self.slice_LR[axis][0]]
            primitives_R = primitives[self.slice_LR[axis][1]]
            gamma_1 = gamma_234 = gamma_5 = jnp.maximum(
                jnp.abs(primitives_L[axis+1]) + self.material_manager.get_speed_of_sound(primitives_L), 
                jnp.abs(primitives_R[axis+1]) + self.material_manager.get_speed_of_sound(primitives_R))

        # GLF EIGENVALUES
        elif self.flux_splitting == "GLF":
            primitives = primitives[self.slice_domain_plus_one[axis]]
            gamma_12345 = jnp.max(jnp.abs(primitives[axis+1]) + self.material_manager.get_speed_of_sound(primitives))
            gamma_1 = gamma_234 = gamma_5 = gamma_12345 * jnp.ones(_s)

        else:
            raise NotImplementedError

        #TODO this is not nice 
        eigvals = jnp.array([
            [gamma_1      , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s), jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_234    , jnp.zeros(_s)],
            [jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), jnp.zeros(_s), gamma_5      ],
        ])

        return right_eigs, left_eigs, eigvals    

    def transformtochar(self, stencil: Array, left_eig: Array, axis: int) -> Array:
        """Transforms the stencil from physical to characteristic space.

        :param stencil: Buffer with variables in physical space.
        :type stencil: Array
        :param left_eig: Buffer of left eigenvalues.
        :type left_eig: Array
        :param axis: Spatial direction along which transformation has to be performed. 
        :type axis: int
        :return: Buffer with variables in characteristic space.
        :rtype: Array
        """
        left_eig = jnp.expand_dims(left_eig, axis=axis-3)
        return jnp.einsum("ij...,j...->i...", left_eig, stencil)

    def transformtophysical(self, stencil: Array, right_eig: Array) -> Array:
        """Transforms the stencil from characteristic to physical space.

        :param stencil: Buffer with variables in characteristic space.
        :type stencil: Array
        :param right_eig: Buffer of right eigenvalues.
        :type right_eig: Array
        :return: Buffer with variables in physical space.
        :rtype: Array
        """
        return jnp.einsum("ij...,j...->i...", right_eig, stencil)
    

    def get_amplitude_variations_from_primitives(self, dq_dx: Array, primitives: Array, axis: int) -> Array:

        speed_of_sound = self.material_manager.get_speed_of_sound(primitives)

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            vel_id_major = self.vel_ids[axis]
            vel_id_minor_0 = self.vel_minors[axis][0]
            vel_id_minor_1 = self.vel_minors[axis][1]

            lambda_min = primitives[vel_id_major] - speed_of_sound
            lambda_mid = primitives[vel_id_major]
            lambda_max = primitives[vel_id_major] + speed_of_sound

            L_waves = [0 for _ in range(self.equation_information.no_primes)]
        
            L_waves[self.ids_mass]              = lambda_min * (-speed_of_sound * primitives[self.ids_mass] * dq_dx[vel_id_major] + dq_dx[self.ids_energy])
            L_waves[1]                          = lambda_mid * (jnp.square(speed_of_sound) * dq_dx[self.ids_mass] - dq_dx[self.ids_energy])
            L_waves[self.vel_id_eigs[axis][0]]  = lambda_mid * dq_dx[vel_id_minor_0]
            L_waves[self.vel_id_eigs[axis][1]]  = lambda_mid * dq_dx[vel_id_minor_1]
            L_waves[self.ids_energy]            = lambda_max * (speed_of_sound * primitives[self.ids_mass] * dq_dx[vel_id_major] + dq_dx[self.ids_energy])
            
            L_waves = jnp.array(L_waves)

        else:
            raise NotImplementedError
        
        return L_waves

    def get_primitives_from_amplitude_variations(self, L_waves: Array, primitives: Array, axis: int) -> Array:

        speed_of_sound = self.material_manager.get_speed_of_sound(primitives)

        if self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):

            vel_id_major = self.vel_ids[axis]
            vel_id_minor_0 = self.vel_minors[axis][0]
            vel_id_minor_1 = self.vel_minors[axis][1]

            one_speed_of_sound = 1.0 / speed_of_sound
            one_speed_of_sound_squared = jnp.square(one_speed_of_sound)

            one_lambda_min = 1.0 / (primitives[vel_id_major] - speed_of_sound)
            one_lambda_mid = 1.0 / (primitives[vel_id_major] + 1e-10)
            one_lambda_max = 1.0 / (primitives[vel_id_major] + speed_of_sound)

            dq_dx = [0 for _ in range(self.equation_information.no_primes)]

            dq_dx[self.ids_mass] = one_speed_of_sound_squared * (
                0.5 * one_lambda_min * L_waves[0] + one_lambda_mid * L_waves[1] + 0.5 * one_lambda_max * L_waves[4])
            dq_dx[vel_id_major] = 0.5 * one_speed_of_sound / primitives[self.ids_mass] * (
                -one_lambda_min * L_waves[0] + one_lambda_max * L_waves[4])
            dq_dx[self.vel_minors[axis][0]] = one_lambda_mid * L_waves[self.vel_id_eigs[axis][0]]
            dq_dx[self.vel_minors[axis][1]] = one_lambda_mid * L_waves[self.vel_id_eigs[axis][1]]
            dq_dx[self.ids_energy] = 0.5 * (one_lambda_min * L_waves[0] + one_lambda_max * L_waves[4])

            dq_dx = jnp.array(dq_dx)

        else:
            raise NotImplementedError

        return dq_dx
 