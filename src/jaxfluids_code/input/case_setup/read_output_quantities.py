from typing import Dict
import warnings

from jaxfluids.data_types.case_setup.output import OutputQuantitiesSetup
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.equation_information import AVAILABLE_QUANTITIES
from jaxfluids.input.case_setup import get_path_to_key, get_setup_value
from jaxfluids.input.setup_reader import assert_case

def read_output_quantities(
        case_setup_dict: Dict,
        numerical_setup: NumericalSetup
        ) -> OutputQuantitiesSetup:

    basepath = get_path_to_key("output", "quantities")

    output_dict = get_setup_value(
        case_setup_dict, "output", basepath, dict,
        is_optional=True, default_value={})
    
    path = get_path_to_key(basepath, "primitives")
    primitives = get_setup_value(
        output_dict, "primitives", path, list,
        is_optional=True, default_value=None)

    path = get_path_to_key(basepath, "conservatives")
    conservatives = get_setup_value(
        output_dict, "conservatives", path, list,
        is_optional=True, default_value=None)

    path = get_path_to_key(basepath, "real_fluid")
    real_fluid = get_setup_value(
        output_dict, "real_fluid", path, list,
        is_optional=True, default_value=None)

    path = get_path_to_key(basepath, "levelset")
    levelset = get_setup_value(
        output_dict, "levelset", path, list,
        is_optional=True, default_value=None)
    
    path = get_path_to_key(basepath, "solids")
    solids = get_setup_value(
        output_dict, "solids", path, list,
        is_optional=True, default_value=None)
    
    path = get_path_to_key(basepath, "miscellaneous")
    miscellaneous = get_setup_value(
        output_dict, "miscellaneous", path, list,
        is_optional=True, default_value=None)
    
    path = get_path_to_key(basepath, "forcings")
    forcings = get_setup_value(
        output_dict, "forcings", path, list,
        is_optional=True, default_value=None)

    output_quantities_setup = OutputQuantitiesSetup(   
        primitives, conservatives, real_fluid,
        levelset, solids, miscellaneous,
        forcings)

    sanity_check(output_quantities_setup, numerical_setup)

    return output_quantities_setup

def sanity_check(
        output_quantities_setup: OutputQuantitiesSetup,
        numerical_setup: NumericalSetup
        ) -> None:

    is_all_output_quantities_none = all(map(lambda x: x is None, output_quantities_setup))
    if is_all_output_quantities_none:
        warning_str = ("Currently all output quantities are None. "
                       "The simulation will run but no output will be written.")
        warnings.warn(warning_str, RuntimeWarning)

    for field in output_quantities_setup._fields:
        quantities = getattr(output_quantities_setup, field)
        if quantities != None:
            cond = len(quantities) == len(set(quantities))
            assert_str = f"Output field '{field:s}' contains duplicates."
            assert_case(cond, assert_str)

            for quantity in quantities:
                cond = quantity in AVAILABLE_QUANTITIES[field]
                assert_str = f"Output quantity '{field:s}/{quantity:s}' does not exist."
                assert_case(cond, assert_str)

    if numerical_setup.levelset.model != "FLUID-FLUID":
        real_fluid = getattr(output_quantities_setup, "real_fluid")
        if real_fluid != None:
            cond = len(real_fluid) == 0
            assert_str = "Real fluid output not available with given levelset model."
            assert_case(cond, assert_str)

