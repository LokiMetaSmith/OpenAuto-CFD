"""
viewer/project_engine.py

Unified Multiphysics Project Management Architecture.
Discovers, loads, and orchestrates cross-disciplinary hardware projects combining:
  1. Multi-board KiCad ECAD layouts (.kicad_pcb, .kicad_sch, stackups)
  2. Microfluidic & Mechanical CAD (OpenSCAD, STL wafer molds, housings)
  3. Multiphysics Co-Simulation & Physical Coupling parameters
"""

import os
import sys
import json
import glob
import re
from typing import Dict, Any, List, Optional

KICAD_PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kicad_plugin"))
if KICAD_PLUGIN_DIR not in sys.path:
    sys.path.insert(0, KICAD_PLUGIN_DIR)

from em_live_watcher import EMLiveSyncDaemon
from eda_rf_driver import KiCadPcbExporter

try:
    from samara_hardware_driver import SamaraHardwareDriver
except ImportError:
    try:
        from optimizer.samara_hardware_driver import SamaraHardwareDriver
    except ImportError:
        SamaraHardwareDriver = None


def get_default_project_search_dirs() -> List[str]:
    """
    Dynamically identifies project search directories across any machine or OS.
    Checks:
    1. PROJECTS_DIR / OPENAUTO_PROJECTS_DIR environment variables
    2. User's ~/Documents/projects and ~/projects
    3. Parent directory of this repository (peer repos)
    4. Current repository directory
    5. Fallback paths if existing
    """
    dirs = []
    # 1. Environment variables
    for env_var in ["PROJECTS_DIR", "OPENAUTO_PROJECTS_DIR"]:
        val = os.environ.get(env_var)
        if val and os.path.isdir(val):
            abs_p = os.path.abspath(val)
            if abs_p not in dirs:
                dirs.append(abs_p)

    # 2. User home directories
    try:
        home = os.path.expanduser("~")
        for cand in [
            os.path.join(home, "Documents", "projects"),
            os.path.join(home, "projects"),
            os.path.join(home, "Documents"),
        ]:
            if os.path.isdir(cand):
                abs_p = os.path.abspath(cand)
                if abs_p not in dirs:
                    dirs.append(abs_p)
    except Exception:
        pass

    # 3. Known development paths (if existing on current machine)
    for p in [r"C:\Users\Loki-VR\Documents\projects"]:
        if os.path.isdir(p):
            abs_p = os.path.abspath(p)
            if abs_p not in dirs:
                dirs.append(abs_p)

    # 4. Parent directory of this repository (e.g. peer repositories)
    repo_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.isdir(repo_parent) and repo_parent not in dirs:
        dirs.append(repo_parent)

    # 5. This repository
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(repo_dir) and repo_dir not in dirs:
        dirs.append(repo_dir)

    return dirs


class MultiphysicsProject:
    """Represents a validated multi-disciplinary hardware project."""

    def __init__(self, manifest_path: str):
        self.manifest_path = os.path.abspath(manifest_path)
        self.project_dir = os.path.dirname(self.manifest_path)
        self.raw_manifest = self._load_manifest()

        self.project_id: str = self.raw_manifest.get("project_id", os.path.basename(self.project_dir))
        self.name: str = self.raw_manifest.get("name", self.project_id)
        self.version: str = self.raw_manifest.get("version", "1.0.0")
        self.description: str = self.raw_manifest.get("description", "")
        self.active_board_id: str = self.raw_manifest.get("active_board", "")
        self.fluidics: Dict[str, Any] = self.raw_manifest.get("fluidics", {})
        self.cosimulation: Dict[str, Any] = self.raw_manifest.get("cosimulation", {})

        # Resolve Boards
        self.boards: Dict[str, Dict[str, Any]] = {}
        raw_boards = self.raw_manifest.get("boards", {})
        for bid, binfo in raw_boards.items():
            pcb_rel = binfo.get("path", "")
            sch_rel = binfo.get("schematic_path", "")
            pcb_abs = os.path.abspath(os.path.join(self.project_dir, pcb_rel)) if pcb_rel else None
            sch_abs = os.path.abspath(os.path.join(self.project_dir, sch_rel)) if sch_rel else None

            self.boards[bid] = {
                "id": bid,
                "name": binfo.get("name", bid),
                "role": binfo.get("role", "PCB Layout"),
                "path": pcb_abs,
                "relative_path": pcb_rel,
                "exists": os.path.exists(pcb_abs) if pcb_abs else False,
                "schematic_path": sch_abs,
                "substrate": binfo.get("substrate", {}),
                "spice": binfo.get("spice", {}),
                "fluidic_coupling": binfo.get("fluidic_coupling", {})
            }

        # Default active board if invalid or empty
        if self.active_board_id not in self.boards and self.boards:
            self.active_board_id = next(iter(self.boards.keys()))

        # Resolve Mechanical Parts / CAD
        self.mechanical: Dict[str, Any] = self.raw_manifest.get("mechanical", {})
        parts = self.mechanical.get("parts", []) or self.mechanical.get("models", [])
        resolved_parts = []
        for p in parts:
            p_rel = p.get("path", "")
            p_abs = os.path.abspath(os.path.join(self.project_dir, p_rel)) if p_rel else None
            resolved_parts.append({
                "id": p.get("id", os.path.splitext(os.path.basename(p_rel))[0]),
                "name": p.get("name", os.path.basename(p_rel)),
                "type": p.get("type", "part"),
                "path": p_abs,
                "relative_path": p_rel,
                "exists": os.path.exists(p_abs) if p_abs else False,
                "color": p.get("color", "#60a5fa")
            })
        if not resolved_parts and os.path.isdir(self.project_dir):
            resolved_parts = self._auto_discover_mechanical_parts()

        self.mechanical_parts = resolved_parts

    def _auto_discover_mechanical_parts(self) -> List[Dict[str, Any]]:
        """Automatically scans project directory for 3D CAD files (.stl, .step)."""
        discovered = []
        palette = ["#60a5fa", "#34d399", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]
        cand_patterns = [
            os.path.join(self.project_dir, "*.stl"),
            os.path.join(self.project_dir, "*.step"),
            os.path.join(self.project_dir, "build123d_output", "*.stl"),
            os.path.join(self.project_dir, "build123d_output", "*.step"),
            os.path.join(self.project_dir, "cad", "*.stl"),
            os.path.join(self.project_dir, "cadquery_output", "*.stl"),
        ]
        seen_files = set()
        for pat in cand_patterns:
            for f in glob.glob(pat):
                abs_f = os.path.abspath(f)
                if abs_f in seen_files:
                    continue
                seen_files.add(abs_f)
                rel_f = os.path.relpath(abs_f, self.project_dir).replace("\\", "/")
                base_name = os.path.splitext(os.path.basename(abs_f))[0]
                part_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name.lower())
                color = palette[len(discovered) % len(palette)]
                discovered.append({
                    "id": part_id,
                    "name": base_name.replace("_", " ").title(),
                    "type": "stl" if abs_f.lower().endswith(".stl") else "cad",
                    "path": abs_f,
                    "relative_path": rel_f,
                    "exists": True,
                    "color": color
                })
        return discovered

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[MultiphysicsProject] Error reading {self.manifest_path}: {e}")
            return {}

    def get_active_board(self) -> Optional[Dict[str, Any]]:
        return self.boards.get(self.active_board_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "project_dir": self.project_dir,
            "manifest_path": self.manifest_path,
            "active_board_id": self.active_board_id,
            "boards": self.boards,
            "mechanical": {
                "cad_format": self.mechanical.get("cad_format", "STL"),
                "parts": self.mechanical_parts
            },
            "fluidics": self.fluidics,
            "cosimulation": self.cosimulation
        }


class ProjectManager:
    """
    Central discovery, registry, and switching manager for projects and boards.
    """

    DEFAULT_PROJECT_SEARCH_DIRS = get_default_project_search_dirs()

    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url.rstrip("/")
        self.search_dirs: List[str] = list(get_default_project_search_dirs())
        self.projects: Dict[str, MultiphysicsProject] = {}
        self.active_project_id: Optional[str] = None
        self.active_daemon: Optional[EMLiveSyncDaemon] = None
        self.on_state_change_callback = None

        self.discover_projects()

        # Prioritize Corkscrew Filter or first discovered project, then Daemon Pore
        if "corkscrew-filter" in self.projects:
            self.active_project_id = "corkscrew-filter"
        elif "daemon-pore" in self.projects:
            self.active_project_id = "daemon-pore"
        elif self.projects:
            self.active_project_id = next(iter(self.projects.keys()))

    def add_search_dir(self, directory: str) -> bool:
        """Adds a new directory to the project discovery path."""
        if not directory or not os.path.isdir(directory):
            return False
        abs_dir = os.path.abspath(directory)
        if abs_dir not in self.search_dirs:
            self.search_dirs.insert(0, abs_dir)
            self.discover_projects()
            return True
        return False

    def discover_projects(self, additional_dirs: Optional[List[str]] = None) -> Dict[str, MultiphysicsProject]:
        """Scan candidate directories for openauto.project.json (or fallback atlas.project.json) files."""
        found = {}
        seen_dirs = set()

        all_dirs = list(self.search_dirs)
        if additional_dirs:
            all_dirs.extend(additional_dirs)

        for search_dir in all_dirs:
            if not os.path.exists(search_dir):
                continue
            abs_search = os.path.abspath(search_dir)
            if abs_search in seen_dirs:
                continue
            seen_dirs.add(abs_search)

            # Look for openauto.project.json or atlas.project.json up to 3 levels deep
            manifest_names = ["openauto.project.json", "atlas.project.json"]
            for manifest_name in manifest_names:
                patterns = [
                    os.path.join(abs_search, manifest_name),
                    os.path.join(abs_search, "*", manifest_name),
                    os.path.join(abs_search, "*", "*", manifest_name),
                ]
                for pat in patterns:
                    for manifest_file in glob.glob(pat):
                        try:
                            proj = MultiphysicsProject(manifest_file)
                            # Do not overwrite if openauto.project.json already loaded for this project
                            if proj.project_id not in found or manifest_name == "openauto.project.json":
                                found[proj.project_id] = proj
                        except Exception as e:
                            print(f"[ProjectManager] Failed to load {manifest_file}: {e}")

        self.projects = found
        return self.projects

    def get_active_project(self) -> Optional[MultiphysicsProject]:
        if not self.active_project_id or self.active_project_id not in self.projects:
            if self.projects:
                self.active_project_id = next(iter(self.projects.keys()))
            else:
                return None
        return self.projects[self.active_project_id]

    def list_projects(self) -> List[Dict[str, Any]]:
        """Returns concise list of projects for UI dropdown."""
        self.discover_projects()
        res = []
        for pid, proj in self.projects.items():
            res.append({
                "project_id": pid,
                "name": proj.name,
                "version": proj.version,
                "description": proj.description,
                "project_dir": proj.project_dir,
                "boards_count": len(proj.boards),
                "mechanical_count": len(proj.mechanical_parts),
                "is_active": pid == self.active_project_id,
                "active_board_id": proj.active_board_id,
                "boards": [
                    {
                        "id": bid,
                        "name": binfo["name"],
                        "role": binfo["role"],
                        "exists": binfo["exists"]
                    }
                    for bid, binfo in proj.boards.items()
                ]
            })
        return res

    def select_project(self, project_id: str) -> Dict[str, Any]:
        """Switches active project."""
        if project_id not in self.projects:
            # Re-discover in case a new project was created
            self.discover_projects()

        if project_id not in self.projects:
            raise ValueError(f"Project '{project_id}' not found. Available: {list(self.projects.keys())}")

        self.active_project_id = project_id
        proj = self.projects[project_id]

        # Sync active board
        board_sync_data = self.trigger_active_board_sync()

        return {
            "success": True,
            "active_project": proj.to_dict(),
            "board_sync": board_sync_data
        }

    def select_board(self, board_id: str) -> Dict[str, Any]:
        """Switches active board within active project."""
        proj = self.get_active_project()
        if not proj:
            raise ValueError("No active project selected")

        if board_id not in proj.boards:
            raise ValueError(f"Board '{board_id}' not found in project '{proj.project_id}'. Available: {list(proj.boards.keys())}")

        proj.active_board_id = board_id
        board_info = proj.boards[board_id]

        board_sync_data = self.trigger_active_board_sync()

        return {
            "success": True,
            "active_project_id": proj.project_id,
            "active_board_id": board_id,
            "board_info": board_info,
            "board_sync": board_sync_data
        }

    def get_active_board_path(self) -> Optional[str]:
        proj = self.get_active_project()
        if not proj:
            return None
        b = proj.get_active_board()
        if b and b.get("path") and os.path.exists(b["path"]):
            return b["path"]
        return None

    def trigger_active_board_sync(self) -> Optional[Dict[str, Any]]:
        """Parses active board file, stops old daemon, and starts new daemon on active board."""
        board_path = self.get_active_board_path()
        if not board_path:
            return None

        # Stop existing watcher if pointing to different file
        if self.active_daemon:
            if self.active_daemon.pcb_filepath != os.path.abspath(board_path):
                self.active_daemon.stop()
                self.active_daemon = None

        # Create or update daemon
        if not self.active_daemon:
            self.active_daemon = EMLiveSyncDaemon(
                pcb_filepath=board_path,
                server_url=self.server_url
            )
            # Daemon watcher thread
            self.active_daemon.start()

        # Execute immediate sync calculation
        sync_payload = self.active_daemon.trigger_sync()

        # Augment payload with project & board metadata
        if sync_payload:
            proj = self.get_active_project()
            if proj:
                sync_payload["project_id"] = proj.project_id
                sync_payload["project_name"] = proj.name
                sync_payload["board_id"] = proj.active_board_id
                board_info = proj.get_active_board() or {}
                sync_payload["board_role"] = board_info.get("role", "")
                sync_payload["mechanical_parts"] = proj.mechanical_parts
                sync_payload["fluidics"] = proj.fluidics
                sync_payload["cosimulation"] = proj.cosimulation

                # If board has custom substrate specs in manifest, apply them
                sub = board_info.get("substrate", {})
                if sub and "stackup" in sync_payload:
                    if "er" in sub:
                        sync_payload["stackup"]["dielectric_constant"] = float(sub["er"])
                    if "thickness_mm" in sub:
                        sync_payload["stackup"]["substrate_height_mm"] = float(sub["thickness_mm"])
                    if "material" in sub:
                        sync_payload["stackup"]["material_name"] = str(sub["material"])

        return sync_payload

    def create_project(
        self,
        name: str,
        project_id: Optional[str] = None,
        project_dir: Optional[str] = None,
        description: str = "",
        board_name: str = "Main PCB",
        board_filename: str = "board.kicad_pcb",
        substrate_material: str = "FR4 High-TG",
        substrate_er: float = 4.3,
        substrate_thickness_mm: float = 1.6,
        mechanical_parts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Creates a new multiphysics project, synthesizes openauto.project.json, generates starter PCB if needed, and activates it."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty")

        if not project_id:
            project_id = re.sub(r'[^a-zA-Z0-9_-]', '-', clean_name.lower()).strip('-')
            if not project_id:
                project_id = "project"

        # Unique slug check
        base_id = project_id
        counter = 1
        while project_id in self.projects:
            project_id = f"{base_id}-{counter}"
            counter += 1

        # Project Directory
        if not project_dir or not project_dir.strip():
            base_dir = self.search_dirs[0] if self.search_dirs else os.path.expanduser("~/Documents/projects")
            os.makedirs(base_dir, exist_ok=True)
            project_dir = os.path.join(base_dir, clean_name)
        project_dir = os.path.abspath(project_dir.strip())
        os.makedirs(project_dir, exist_ok=True)

        # Primary Board resolution
        if not board_filename or not board_filename.strip():
            board_filename = "board.kicad_pcb"
        board_clean = board_filename.strip()
        if not board_clean.endswith(".kicad_pcb"):
            board_clean += ".kicad_pcb"

        if os.path.isabs(board_clean):
            board_abs = board_clean
            board_rel = os.path.relpath(board_clean, project_dir)
        else:
            board_rel = board_clean
            board_abs = os.path.abspath(os.path.join(project_dir, board_clean))

        # If board does not exist, synthesize a starter KiCad PCB layout
        if not os.path.exists(board_abs):
            try:
                KiCadPcbExporter.generate_kicad_pcb(
                    trace_width_mm=0.35,
                    line_length_mm=40.0,
                    board_width_mm=35.0,
                    board_length_mm=55.0,
                    output_filepath=board_abs
                )
            except Exception as e:
                print(f"[ProjectManager] Notice: Could not generate starter KiCad PCB: {e}")

        # Auto-discover mechanical parts if none provided
        if not mechanical_parts and os.path.isdir(project_dir):
            palette = ["#60a5fa", "#34d399", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]
            cand_patterns = [
                os.path.join(project_dir, "*.stl"),
                os.path.join(project_dir, "*.step"),
                os.path.join(project_dir, "build123d_output", "*.stl"),
                os.path.join(project_dir, "build123d_output", "*.step"),
                os.path.join(project_dir, "cad", "*.stl"),
                os.path.join(project_dir, "cadquery_output", "*.stl"),
            ]
            discovered_parts = []
            seen_files = set()
            for pat in cand_patterns:
                for f in glob.glob(pat):
                    abs_f = os.path.abspath(f)
                    if abs_f in seen_files:
                        continue
                    seen_files.add(abs_f)
                    rel_f = os.path.relpath(abs_f, project_dir).replace("\\", "/")
                    base_n = os.path.splitext(os.path.basename(abs_f))[0]
                    pid = re.sub(r'[^a-zA-Z0-9_-]', '_', base_n.lower())
                    color = palette[len(discovered_parts) % len(palette)]
                    discovered_parts.append({
                        "id": pid,
                        "name": base_n.replace("_", " ").title(),
                        "path": rel_f,
                        "type": "stl" if abs_f.lower().endswith(".stl") else "step",
                        "color": color
                    })
            mechanical_parts = discovered_parts

        # Construct openauto.project.json manifest
        manifest_data = {
            "project_id": project_id,
            "name": clean_name,
            "version": "1.0.0",
            "description": description.strip() if description else f"{clean_name} Multiphysics Project",
            "root_path": ".",
            "active_board": "main",
            "boards": {
                "main": {
                    "name": board_name.strip() or "Main PCB",
                    "path": board_rel.replace("\\", "/"),
                    "role": "Primary System Board",
                    "substrate": {
                        "material": substrate_material,
                        "er": float(substrate_er),
                        "thickness_mm": float(substrate_thickness_mm)
                    }
                }
            },
            "mechanical": {
                "cad_format": "OpenSCAD / STL",
                "parts": mechanical_parts or []
            },
            "fluidics": {},
            "cosimulation": {
                "enabled": True
            }
        }

        manifest_path = os.path.join(project_dir, "openauto.project.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Re-discover projects and select newly created project
        self.add_search_dir(project_dir)
        self.discover_projects()
        switch_res = self.select_project(project_id)

        return {
            "success": True,
            "project_id": project_id,
            "project_dir": project_dir,
            "manifest_path": manifest_path,
            "active_project": self.projects[project_id].to_dict(),
            "board_sync": switch_res.get("board_sync")
        }

    def import_project(
        self,
        project_dir: str,
        name: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Imports an external directory into OpenAuto-CFD.
        If openauto.project.json (or atlas.project.json) does not exist:
          - Automatically scans for CAD parts (.stl, .step, .scad)
          - Automatically scans for KiCad boards or footprints (.kicad_pcb, .kicad_mod)
          - If a footprint or generator exists (e.g. Samara planar stator), synthesizes a board
          - Generates openauto.project.json
        Registers the project directory in search dirs, rediscovers, and activates it.
        """
        if not project_dir or not os.path.exists(project_dir):
            raise ValueError(f"Project directory not found: '{project_dir}'")

        project_dir = os.path.abspath(project_dir)
        dir_basename = os.path.basename(project_dir)

        # Detect project name
        if not name:
            readme_p = os.path.join(project_dir, "README.md")
            if os.path.exists(readme_p):
                try:
                    with open(readme_p, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        if first_line.startswith("#"):
                            name = first_line.lstrip("#").split(":")[0].strip()
                except Exception:
                    pass
        if not name:
            name = dir_basename.replace("_", " ").replace("-", " ").title()

        if not project_id:
            project_id = re.sub(r'[^a-zA-Z0-9_-]', '-', name.lower()).strip('-')
            if not project_id:
                project_id = dir_basename.lower()

        # Check for existing manifest
        manifest_names = ["openauto.project.json", "atlas.project.json"]
        existing_manifest = None
        for mf in manifest_names:
            p = os.path.join(project_dir, mf)
            if os.path.exists(p):
                existing_manifest = p
                break

        if not existing_manifest:
            # Auto-discover CAD parts
            palette = ["#60a5fa", "#34d399", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]
            cand_patterns = [
                os.path.join(project_dir, "*.stl"),
                os.path.join(project_dir, "*.step"),
                os.path.join(project_dir, "build123d_output", "*.stl"),
                os.path.join(project_dir, "build123d_output", "*.step"),
                os.path.join(project_dir, "cad", "*.stl"),
                os.path.join(project_dir, "cadquery_output", "*.stl"),
            ]
            discovered_parts = []
            seen_files = set()
            for pat in cand_patterns:
                for f in glob.glob(pat):
                    abs_f = os.path.abspath(f)
                    if abs_f in seen_files:
                        continue
                    seen_files.add(abs_f)
                    rel_f = os.path.relpath(abs_f, project_dir).replace("\\", "/")
                    base_n = os.path.splitext(os.path.basename(abs_f))[0]
                    pid = re.sub(r'[^a-zA-Z0-9_-]', '_', base_n.lower())
                    color = palette[len(discovered_parts) % len(palette)]
                    discovered_parts.append({
                        "id": pid,
                        "name": base_n.replace("_", " ").title(),
                        "path": rel_f,
                        "type": "stl" if abs_f.lower().endswith(".stl") else "step",
                        "color": color
                    })

            # Auto-discover or synthesize KiCad boards
            boards_dict = {}
            kicad_pcb_files = glob.glob(os.path.join(project_dir, "*.kicad_pcb")) + \
                              glob.glob(os.path.join(project_dir, "hardware", "*.kicad_pcb"))

            if kicad_pcb_files:
                for b_file in kicad_pcb_files:
                    b_rel = os.path.relpath(b_file, project_dir).replace("\\", "/")
                    b_id = os.path.splitext(os.path.basename(b_file))[0].lower().replace("_", "-")
                    boards_dict[b_id] = {
                        "name": b_id.replace("-", " ").title(),
                        "path": b_rel,
                        "role": "System Board",
                        "substrate": {"material": "FR4 High-TG", "er": 4.3, "thickness_mm": 1.6}
                    }
            else:
                # Check if Samara planar stator footprint exists
                stator_mod = os.path.join(project_dir, "hardware", "planar_stator.kicad_mod")
                if os.path.exists(stator_mod) or "samara" in project_id:
                    stator_pcb_abs = os.path.join(project_dir, "hardware", "samara_stator.kicad_pcb")
                    if SamaraHardwareDriver:
                        driver = SamaraHardwareDriver(project_dir)
                        driver.generate_samara_stator_pcb(stator_pcb_abs)
                    else:
                        KiCadPcbExporter.generate_kicad_pcb(0.25, 40.0, 40.0, 40.0, output_filepath=stator_pcb_abs)
                    boards_dict["stator"] = {
                        "name": "Planar Stator PCB",
                        "path": "hardware/samara_stator.kicad_pcb",
                        "role": "Multiphase Stator & Sensor PCB",
                        "substrate": {"material": "FR4 High-TG", "er": 4.3, "thickness_mm": 1.6}
                    }
                else:
                    starter_pcb = os.path.join(project_dir, "hardware", "board.kicad_pcb")
                    KiCadPcbExporter.generate_kicad_pcb(0.35, 40.0, 35.0, 55.0, output_filepath=starter_pcb)
                    boards_dict["main"] = {
                        "name": "Main System PCB",
                        "path": "hardware/board.kicad_pcb",
                        "role": "System Board",
                        "substrate": {"material": "FR4 High-TG", "er": 4.3, "thickness_mm": 1.6}
                    }

            active_b_id = next(iter(boards_dict.keys())) if boards_dict else "main"

            manifest_data = {
                "project_id": project_id,
                "name": name,
                "version": "1.0.0",
                "description": f"{name} Multiphysics Project",
                "root_path": ".",
                "active_board": active_b_id,
                "boards": boards_dict,
                "mechanical": {
                    "cad_format": "OpenSCAD / STL",
                    "parts": discovered_parts
                },
                "fluidics": {},
                "cosimulation": {
                    "enabled": True
                }
            }

            manifest_path = os.path.join(project_dir, "openauto.project.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

        # Register search directory and reload
        self.add_search_dir(project_dir)
        parent_dir = os.path.dirname(project_dir)
        if parent_dir:
            self.add_search_dir(parent_dir)

        self.discover_projects()

        matched_id = None
        for pid, proj in self.projects.items():
            if os.path.abspath(proj.project_dir) == project_dir:
                matched_id = pid
                break

        if not matched_id:
            matched_id = project_id
            if matched_id not in self.projects:
                mf_file = os.path.join(project_dir, "openauto.project.json")
                if os.path.exists(mf_file):
                    self.projects[matched_id] = MultiphysicsProject(mf_file)

        switch_res = self.select_project(matched_id)
        return {
            "success": True,
            "project_id": matched_id,
            "project_dir": project_dir,
            "active_project": self.projects[matched_id].to_dict(),
            "board_sync": switch_res.get("board_sync")
        }
