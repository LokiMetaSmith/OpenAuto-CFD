import os
from physics_driver import PhysicsDriver

class JointPhysicsDriver(PhysicsDriver):
    """
    A composite driver that sequentially executes multiple physics simulations.
    Currently hardcoded to execute both FoamDriver (CFD) and OpenEMSDriver (EM).
    """

    def __init__(self, case_dir, config=None, container_engine="auto", verbose=False, debug=False, **kwargs):
        super().__init__(case_dir, config=config, container_engine=container_engine, verbose=verbose, debug=debug)

        # Instantiate both drivers
        from foam_driver import FoamDriver
        from em_driver import OpenEMSDriver

        # Create sub-directories for each solver to avoid file conflicts
        cfd_case_dir = os.path.join(self.case_dir, "cfd")
        em_case_dir = os.path.join(self.case_dir, "em")

        # When creating the sub-drivers, we point their template directories to the parent template
        # directory so they can find 0.orig, constant, system, etc.
        template_dir = kwargs.get('template_dir')
        cfd_template = template_dir if template_dir else case_dir
        em_template = template_dir if template_dir else case_dir

        # Copy kwargs and set template_dir
        cfd_kwargs = kwargs.copy()
        cfd_kwargs['template_dir'] = cfd_template
        em_kwargs = kwargs.copy()
        em_kwargs['template_dir'] = em_template

        self.cfd_driver = FoamDriver(cfd_case_dir, config=config, container_engine=container_engine, verbose=verbose, debug=debug, **cfd_kwargs)
        self.em_driver = OpenEMSDriver(em_case_dir, config=config, container_engine=container_engine, verbose=verbose, debug=debug, **em_kwargs)

        self.drivers = [self.cfd_driver, self.em_driver]

    @property
    def has_tools(self):
        # We consider tools present if ALL underlying drivers have their tools
        return all(getattr(d, 'has_tools', False) for d in self.drivers)

    def prepare_case(self, **kwargs):
        """
        Prepares cases for all physics engines.
        """
        for driver in self.drivers:
            print(f"\n--- Preparing Case for {driver.__class__.__name__} ---")
            driver.prepare_case(**kwargs)

    def run_meshing(self, log_file=None, **kwargs):
        """
        Runs meshing sequentially for all physics engines.
        """
        success = True
        for driver in self.drivers:
            print(f"\n--- Running Meshing for {driver.__class__.__name__} ---")
            # Create driver-specific log file
            driver_log = f"{log_file}_{driver.__class__.__name__}" if log_file else None
            if not driver.run_meshing(log_file=driver_log, **kwargs):
                print(f"{driver.__class__.__name__} meshing failed.")
                success = False
        return success

    def run_solver(self, log_file=None, **kwargs):
        """
        Executes solvers sequentially for all physics engines.
        """
        success = True
        for driver in self.drivers:
            print(f"\n--- Running Solver for {driver.__class__.__name__} ---")
            driver_log = f"{log_file}_{driver.__class__.__name__}" if log_file else None
            if not driver.run_solver(log_file=driver_log, **kwargs):
                 print(f"{driver.__class__.__name__} solver failed.")
                 success = False
        return success

    def get_metrics(self, log_file=None):
        """
        Gathers and merges metrics from all physics engines.
        """
        merged_metrics = {}
        for driver in self.drivers:
            driver_log = f"{log_file}_{driver.__class__.__name__}" if log_file else None
            metrics = driver.get_metrics(log_file=driver_log)

            # Prefix errors so they don't overwrite each other if multiple fail
            if "error" in metrics:
                merged_metrics[f"{driver.__class__.__name__}_error"] = metrics.pop("error")

            merged_metrics.update(metrics)

        return merged_metrics

    def cleanup_ram_disk(self):
        """
        Cleans up RAM disks for all physics engines.
        """
        for driver in self.drivers:
            driver.cleanup_ram_disk()

    # Pass-through specific CFD methods required by simulation_runner.py
    # to avoid crashing when the joint driver is invoked
    def update_blockMesh(self, *args, **kwargs):
        self.cfd_driver.update_blockMesh(*args, **kwargs)

    def update_snappyHexMesh_location(self, *args, **kwargs):
        self.cfd_driver.update_snappyHexMesh_location(*args, **kwargs)

    def run_particle_tracking(self, *args, **kwargs):
        self.cfd_driver.run_particle_tracking(*args, **kwargs)

    def generate_vtk(self):
        return self.cfd_driver.generate_vtk()

    def _run_checkMesh(self, *args, **kwargs):
        return self.cfd_driver._run_checkMesh(*args, **kwargs)

    def _classify_mesh(self, *args, **kwargs):
        return self.cfd_driver._classify_mesh(*args, **kwargs)
