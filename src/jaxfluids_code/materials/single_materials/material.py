from abc import ABC, abstractmethod
from typing import Callable, List, Union
import types

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.material_properties import MaterialPropertiesSetup
from jaxfluids.config import precision
from jaxfluids.math.power_functions import power3_2

Array = jax.Array

class Material(ABC):
    """The Material class implements an abstract class
    for a material, i.e., the computation of
    the dynamic viscosity, bulk viscosity and the
    thermal conductivity. The equation of states are
    implemented in the corresonding child classes.
    """

    def __init__(
            self,
            unit_handler: UnitHandler,
            material_setup: MaterialPropertiesSetup,
            **kwargs
            ) -> None:

        self.unit_handler = unit_handler
        self.material_setup = material_setup

        self.eps = precision.get_eps()

        # MATERIAL TYPE SPECIFIC PROPERTIES
        self.molar_mass = None
        self.gamma = None
        self.cp = None
        self.R = None
        self.pb = None
        self.eb = None
        self.et = None


    def _set_transport_properties(self) -> None:
        transport_properties_setup = self.material_setup.transport
        set_champan_enskog = False

        dynamic_viscosity_setup = transport_properties_setup.dynamic_viscosity
        if dynamic_viscosity_setup is not None:
            self.dynamic_viscosity_model = dynamic_viscosity_setup.model
            if self.dynamic_viscosity_model == "CUSTOM":
                self.dynamic_viscosity_fun = dynamic_viscosity_setup.value

            elif self.dynamic_viscosity_model == "SUTHERLAND":
                sutherland_parameters = dynamic_viscosity_setup.sutherland_parameters
                self.mu_ref = sutherland_parameters[0]
                self.T_ref_mu = sutherland_parameters[1]
                self.C_mu = sutherland_parameters[2]
        
            # Add the set up for transport properties for power law here
            if self.dynamic_viscosity_model == "POWERLAW":
                powerlaw_parameters = dynamic_viscosity_setup.powerlaw_parameters
                self.K_ref_mu = powerlaw_parameters[0]
                self.n_ref_mu = powerlaw_parameters[1]

        self.bulk_viscosity = transport_properties_setup.bulk_viscosity

        thermal_conductivity_setup = transport_properties_setup.thermal_conductivity
        if thermal_conductivity_setup is not None:
            self.thermal_conductivity_model = thermal_conductivity_setup.model
            if self.thermal_conductivity_model == "CUSTOM":
                self.thermal_conductivity_fun = thermal_conductivity_setup.value

            elif self.thermal_conductivity_model == "PRANDTL":
                assert_str = ("Consistency error in simulation setup file. Thermal conductivity"
                    " is 'prandtl', however no dynamic viscosity model is specified.")
                assert self.dynamic_viscosity_model is not None, assert_str
                self.prandtl_number = thermal_conductivity_setup.prandtl_number

            elif self.thermal_conductivity_model == "SUTHERLAND":
                sutherland_parameters = thermal_conductivity_setup.sutherland_parameters
                self.kappa_ref = sutherland_parameters[0]
                self.T_ref_kappa = sutherland_parameters[1]
                self.C_kappa = sutherland_parameters[2]

    def get_dynamic_viscosity(self, T: Array, gamma_dot: Array = None) -> Array:
        """Computes the dynamic viscosity.

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

        
        if self.dynamic_viscosity_model == "CUSTOM":
            dynamic_viscosity = self.dynamic_viscosity_fun(T)

        elif self.dynamic_viscosity_model == "SUTHERLAND":
            dynamic_viscosity = \
                self.mu_ref * ((self.T_ref_mu + self.C_mu)/(T + self.C_mu)) * power3_2(T/self.T_ref_mu)

        # Add the power law equation here 
        elif self.dynamic_viscosity_model == "POWERLAW":
            if gamma_dot is None:
                raise ValueError("gamma_dot must be provided for POWERLAW viscosity model.")
            else:
                mu_min = 0.0001
                mu_max = 0.275
                
                from jax import debug
                                
                # Calculate dynamic viscosity using power-law model
                dynamic_viscosity = self.K_ref_mu * jnp.power(gamma_dot, self.n_ref_mu - 1)

                # Clamp dynamic viscosity between mu_inf and mu0 to avoid unphysical values
                dynamic_viscosity = jnp.clip(dynamic_viscosity, mu_min, mu_max)
                #debug.print("dynamic viscosity min: {}, max: {}, any nan: {}", jnp.min(dynamic_viscosity), jnp.max(dynamic_viscosity), jnp.isnan(dynamic_viscosity).any())
        else:
            raise NotImplementedError
        return dynamic_viscosity

    def get_thermal_conductivity(self, T: Array) -> Array:
        """Computes the thermal conductivity

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """
        if self.thermal_conductivity_model == "CUSTOM":
            thermal_conductivity = self.thermal_conductivity_fun(T)

        elif self.thermal_conductivity_model == "PRANDTL":
            thermal_conductivity = self.get_specific_heat_capacity(T) * self.get_dynamic_viscosity(T) / self.prandtl_number

        elif self.thermal_conductivity_model == "SUTHERLAND":
            thermal_conductivity = \
                self.kappa_ref * ((self.T_ref_kappa + self.C_kappa)/(T + self.C_kappa)) * power3_2(T/self.T_ref_kappa)

        else:
            raise NotImplementedError

        return thermal_conductivity

    def get_bulk_viscosity(self, T: Array) -> Array:
        """Computes the bulk viscosity.

        :return: _description_
        :rtype: Array
        """
        return self.bulk_viscosity

    @abstractmethod
    def get_specific_heat_capacity(self, T: Array) -> Array: # TODO clean up inputs for all materials/general cleanup of material module
        """Calculates the specific heat coefficient per unit mass.
        [c_p] = J / kg / K

        :param T: _description_
        :type T: Array
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Array
        """

    @abstractmethod
    def get_speed_of_sound(self, p: Array, rho: Array) -> Array:
        """Computes speed of sound from pressure and density.
        c = c(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Speed of sound buffer
        :rtype: Array
        """

    @abstractmethod
    def get_pressure(self,e: Array, rho: Array) -> Array:
        """Computes pressure from internal energy and density.
        p = p(e, rho)

        :param e: Specific internal energy buffer
        :type e: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Pressue buffer
        :rtype: Array
        """

    @abstractmethod
    def get_temperature(self, p: Array, rho: Array) -> Array:
        """Computes temperature from pressure and density.
        T = T(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Temperature buffer
        :rtype: Array
        """

    @abstractmethod
    def get_density_from_pressure_and_temperature(self, p: Array, T: Array) -> Array:
        pass


    @abstractmethod
    def get_specific_energy(self, p: Array, rho: Array) -> Array:
        """Computes specific internal energy
        e = e(p, rho)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Specific internal energy buffer
        :rtype: Array
        """

    @abstractmethod
    def get_total_energy(
            self,
            p: Array,
            rho: Array,
            velocity_vec: Array,
        ) -> Array:
        """Computes total energy per unit volume from pressure, density, and velocities.
        E = E(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param velocity_vec: Velocity vector, shape = (N_vel,Nx,Ny,Nz)
        :type velocity_vec: Array
        :return: Total energy per unit volume
        :rtype: Array
        """

    @abstractmethod
    def get_total_enthalpy(
            self,
            p: Array,
            rho: Array,
            velocity_vec: Array,
        ) -> Array:
        """Computes total specific enthalpy from pressure, density, and velocities.
        H = H(p, rho, velX, velY, velZ)

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :param velocity_vec: Velocity vector, shape = (N_vel,Nx,Ny,Nz)
        :type velocity_vec: Array
        :return: Total specific enthalpy buffer
        :rtype: Array
        """

    @abstractmethod
    def get_psi(self, p: Array, rho: Array) -> Array:
        """Computes psi from pressure and density.
        psi = p_rho; p_rho is partial derivative of pressure wrt density.

        :param p: Pressure buffer
        :type p: Array
        :param rho: Density buffer
        :type rho: Array
        :return: Psi
        :rtype: Array
        """

    @abstractmethod
    def get_grueneisen(self, rho: Array) -> Array:
        """Computes the Grueneisen coefficient from density.
        Gamma = p_e / rho; p_e is partial derivative of pressure wrt internal specific energy.

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """

    # @abstractmethod
    def get_stagnation_temperature(
            self,
            p:Array,
            rho:Array,
            velocity_vec: Array,
        ) -> Array:
        """Computes the stagnation temperature

        :param rho: Density buffer
        :type rho: Array
        :return: Grueneisen
        :rtype: Array
        """
