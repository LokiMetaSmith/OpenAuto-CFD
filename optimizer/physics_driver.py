class PhysicsDriver:
    """
    Base interface for all physics engine drivers (e.g., OpenFOAM, openEMS).
    """

    def __init__(self, case_dir, config=None, container_engine="auto", verbose=False, debug=False, **kwargs):
        self.case_dir = case_dir
        self.config = config or {}
        self.container_engine = container_engine
        self.verbose = verbose
        self.debug = debug

    def prepare_case(self, **kwargs):
        """
        Prepares the simulation case directory and necessary configuration files.
        """
        raise NotImplementedError

    def run_meshing(self, log_file=None, **kwargs):
        """
        Generates the computational mesh or grid for the simulation.
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError

    def run_solver(self, log_file=None, **kwargs):
        """
        Executes the physics solver.
        Returns True if successful, False otherwise.
        """
        raise NotImplementedError

    def get_metrics(self, log_file=None):
        """
        Parses the simulation output and returns a dictionary of performance metrics.
        """
        raise NotImplementedError

    def cleanup_ram_disk(self):
        """
        Cleans up any temporary RAM disks or volatile storage used by the driver.
        """
        pass
