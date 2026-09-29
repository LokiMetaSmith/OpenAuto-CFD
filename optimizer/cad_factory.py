"""
cad_factory.py
Factory for creating CAD drivers (Build123dDriver vs ScadDriver).
"""

import os
import sys

_opt_dir = os.path.dirname(os.path.abspath(__file__))
if _opt_dir not in sys.path:
    sys.path.insert(0, _opt_dir)

try:
    from scad_driver import ScadDriver
except ImportError:
    try:
        from optimizer.scad_driver import ScadDriver
    except ImportError:
        ScadDriver = None

try:
    from build123d_driver import Build123dDriver
except ImportError:
    try:
        from optimizer.build123d_driver import Build123dDriver
    except ImportError:
        Build123dDriver = None

class CadEngineFactory:
    @staticmethod
    def get_driver(scad_file_path, cad_engine="openscad", fluid_volume_module="modular_filter_assembly", force_native=False):
        """
        Instantiates and returns the selected CAD driver.

        Args:
            scad_file_path (str): Path to CAD file or model identifier.
            cad_engine (str): 'build123d' or 'openscad' / 'scad'.
            fluid_volume_module (str): Name of fluid volume module / part.
            force_native (bool): Force native OpenSCAD CLI if using ScadDriver.

        Returns:
            CadDriverBase: Initialized CAD driver instance.
        """
        if cad_engine == "build123d" and Build123dDriver is not None:
            return Build123dDriver(model_name=scad_file_path, part_name=fluid_volume_module)
        elif ScadDriver is not None:
            return ScadDriver(scad_file_path, fluid_volume_module=fluid_volume_module, force_native=force_native)
        else:
            raise RuntimeError("No CAD engine driver available (neither build123d nor OpenSCAD).")
