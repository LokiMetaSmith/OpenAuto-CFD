import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from optimizer.build123d_driver import Build123dDriver
import build123d as b3d

# Removed xfail since Native python build123d doesn't have the cgal upstream bug openscad has
def test_single_cell_filter_cgal_error(tmp_path):
    """
    Test that reproducing single_cell_filter works without a
    CGAL precondition violation.
    This tracks the upstream issue.
    """
    # We use Build123dDriver instead of native OpenSCAD execution to avoid CGAL errors
    driver = Build123dDriver("corkscrew.scad")
    stl_path = tmp_path / "test_cgal_scf.stl"

    params = {
        "cell_length": 80.0,
        "cell_diameter": 20.0,
        "num_helices": 2,
        "part_to_generate": "single_cell_filter",
        "GENERATE_CFD_VOLUME": False
    }

    success = driver.generate_stl(params, str(stl_path))

    assert success is True
    assert os.path.exists(str(stl_path))

# Removed xfail
def test_flat_end_screw_cgal_error(tmp_path):
    """
    Test that reproducing flat_end_screw works without
    a CGAL NFE / unclosed mesh error.
    """
    driver = Build123dDriver("corkscrew.scad")
    stl_path = tmp_path / "test_cgal_fes.stl"

    params = {
        "part_to_generate": "flat_end_screw",
        "GENERATE_CFD_VOLUME": False
    }

    success = driver.generate_stl(params, str(stl_path))

    assert success is True
    assert os.path.exists(str(stl_path))
