from foam_driver import FoamDriver
from em_driver import OpenEMSDriver
from joint_driver import JointPhysicsDriver

class PhysicsEngineFactory:
    """
    Factory for instantiating the appropriate physics driver based on configuration.
    """

    @staticmethod
    def get_driver(case_dir, config=None, **kwargs):
        """
        Returns an initialized physics driver (e.g., FoamDriver, OpenEMSDriver, or JointPhysicsDriver).

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
        elif physics_type == 'joint':
            return JointPhysicsDriver(case_dir, config=config, **kwargs)
        else:
            raise ValueError(f"Unsupported physics type specified in config: '{physics_type}'. Supported types: 'cfd', 'em', 'joint'.")
