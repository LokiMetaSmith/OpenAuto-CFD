import os
import sys

_opt_dir = os.path.dirname(os.path.abspath(__file__))
if _opt_dir not in sys.path:
    sys.path.insert(0, _opt_dir)

try:
    from foam_driver import FoamDriver
    from em_driver import OpenEMSDriver
    from joint_driver import JointPhysicsDriver
    from fea_driver import FeaDriver
    from meep_driver import MeepDriver
    from s4_driver import S4Driver
except ImportError:
    from optimizer.foam_driver import FoamDriver
    from optimizer.em_driver import OpenEMSDriver
    from optimizer.joint_driver import JointPhysicsDriver
    from optimizer.fea_driver import FeaDriver
    from optimizer.meep_driver import MeepDriver
    from optimizer.s4_driver import S4Driver


class PhysicsEngineFactory:
    """
    Factory for instantiating the appropriate physics driver based on configuration.
    """

    @staticmethod
    def get_driver(case_dir, config=None, **kwargs):
        """
        Returns an initialized physics driver (e.g., FoamDriver, OpenEMSDriver, FeaDriver,
        JointPhysicsDriver, MeepDriver, or S4Driver).

        Args:
            case_dir (str): The path to the case directory.
            config (dict): The configuration dictionary (from YAML).
            **kwargs: Additional arguments to pass to the driver constructor.
        """
        config = config or {}
        physics_type = config.get('physics', {}).get('type', 'cfd').lower()

        if physics_type == 'cfd':
            return FoamDriver(case_dir, config=config, **kwargs)
        elif physics_type == 'em':
            return OpenEMSDriver(case_dir, config=config, **kwargs)
        elif physics_type == 'fea':
            return FeaDriver(case_dir, config=config, **kwargs)
        elif physics_type == 'joint':
            return JointPhysicsDriver(case_dir, config=config, **kwargs)
        elif physics_type in ('meep', 'fdtd'):
            return MeepDriver(case_dir, config=config, **kwargs)
        elif physics_type in ('s4', 'rcwa'):
            return S4Driver(case_dir, config=config, **kwargs)
        elif physics_type in ('photonic', 'optics'):
            solver = config.get('photonic', {}).get('solver', config.get('physics', {}).get('solver', 'meep')).lower()
            if solver in ('s4', 'rcwa'):
                return S4Driver(case_dir, config=config, **kwargs)
            return MeepDriver(case_dir, config=config, **kwargs)
        else:
            raise ValueError(
                f"Unsupported physics type specified in config: '{physics_type}'. "
                f"Supported types: 'cfd', 'em', 'fea', 'joint', 'meep', 's4', 'fdtd', 'rcwa', 'photonic'."
            )

