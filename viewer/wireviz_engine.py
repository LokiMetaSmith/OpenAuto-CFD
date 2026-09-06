"""
viewer/wireviz_engine.py

WireViz Wiring Harness & Fluidic Interconnect Engine for OpenAuto-CFD.
Provides:
1. Automated compilation of wiring and fluidic harnesses into SVG vector diagrams.
2. Extended fluidic tubing, tube fittings, and hardware modeling.
3. Physical Rule Checks (ERC / FRC): loop resistance, IR drop, dead volume, Poiseuille pressure drop.
4. Structured Bill of Materials (BOM) extraction.
5. Multi-format export (SVG, HTML, CSV, PNG).
"""

import os
import re
import math
import tempfile
import pathlib
import csv
import io
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml

try:
    import wireviz.wireviz as wv
    from wireviz.wireviz import parse as wv_parse
    WIREVIZ_AVAILABLE = True
except ImportError:
    WIREVIZ_AVAILABLE = False


# Standard AWG Copper Conductor Resistance (Ohms / meter at 20 C) and Chassis Ampacity
AWG_DATA = {
    18: {"r_per_m": 0.0209, "ampacity_a": 16.0, "diam_mm": 1.024},
    20: {"r_per_m": 0.0333, "ampacity_a": 11.0, "diam_mm": 0.812},
    22: {"r_per_m": 0.0530, "ampacity_a": 7.0,  "diam_mm": 0.644},
    24: {"r_per_m": 0.0842, "ampacity_a": 3.5,  "diam_mm": 0.511},
    26: {"r_per_m": 0.1339, "ampacity_a": 2.2,  "diam_mm": 0.405},
    28: {"r_per_m": 0.2129, "ampacity_a": 1.4,  "diam_mm": 0.321},
    30: {"r_per_m": 0.3385, "ampacity_a": 0.86, "diam_mm": 0.255},
    32: {"r_per_m": 0.5383, "ampacity_a": 0.53, "diam_mm": 0.202},
    34: {"r_per_m": 0.8560, "ampacity_a": 0.33, "diam_mm": 0.160},
    36: {"r_per_m": 1.3609, "ampacity_a": 0.21, "diam_mm": 0.127},
}

# Color aliasing for WireViz compatibility
COLOR_ALIASES = {
    "CY": "TQ",
    "CYAN": "TQ",
    "TURQUOISE": "TQ",
    "NATURAL": "WH",
    "CLEAR": "WH",
    "AMBER": "YE",
    "GOLD": "YE",
    "BLACK": "BK",
    "RED": "RD",
    "BLUE": "BU",
    "GREEN": "GN",
    "YELLOW": "YE",
    "WHITE": "WH",
    "ORANGE": "OG",
    "VIOLET": "VT",
    "PURPLE": "VT",
    "GREY": "GY",
    "GRAY": "GY",
    "BROWN": "BN",
    "PINK": "PK",
}


def parse_length_m(len_val: Any) -> float:
    if len_val is None:
        return 0.1
    if isinstance(len_val, (int, float)):
        return float(len_val)
    s = str(len_val).strip().lower()
    m = re.search(r'([0-9.]+)\s*(m|mm|cm|in|ft)?', s)
    if not m:
        return 0.1
    try:
        val = float(m.group(1))
        unit = m.group(2) or "m"
        if unit == "mm":
            return val / 1000.0
        elif unit == "cm":
            return val / 100.0
        elif unit == "in":
            return val * 0.0254
        elif unit == "ft":
            return val * 0.3048
        return val
    except Exception:
        return 0.1


class WireVizEngine:
    """Core engine for synthesizing, analyzing, and documenting wire and tubing harnesses."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def preprocess_yaml(self, raw_yaml: str) -> str:
        """
        Cleans and normalizes YAML for robust WireViz & Graphviz parsing:
        - Replaces unescaped '&' characters with 'and' to prevent Graphviz XML parser failures.
        - Maps non-standard color codes (CY -> TQ).
        - Normalizes gauge strings for cables.
        """
        text = raw_yaml

        # 1. Escape unescaped ampersands in string values (Graphviz HTML-like label compatibility)
        # Matches '&' not preceded/followed by valid HTML entities
        text = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', 'and', text)

        # 2. Parse YAML dict to perform structured adjustments
        try:
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                return text

            # Normalize colors and gauges in cables
            cables = data.get("cables", {})
            if isinstance(cables, dict):
                for cname, cdata in cables.items():
                    if not isinstance(cdata, dict):
                        continue

                    # Color aliases
                    colors = cdata.get("colors", [])
                    if isinstance(colors, list):
                        cdata["colors"] = [COLOR_ALIASES.get(str(c).upper(), c) for c in colors]

                    # Gauge cleanup (ensure format is 'X.XX unit' or 'XX AWG')
                    gauge = cdata.get("gauge")
                    if gauge is not None:
                        g_str = str(gauge).strip()
                        # If user typed '1/16 OD x 0.020 ID', move to type/notes and simplify gauge
                        if "ID" in g_str or "OD" in g_str or "x" in g_str:
                            if "type" not in cdata:
                                cdata["type"] = f"Microbore Tubing ({g_str})"
                            # Extract first numeric dimension for gauge
                            num_match = re.search(r'([0-9.]+)\s*(mm|in|um)?', g_str)
                            if num_match:
                                val = num_match.group(1)
                                unit = num_match.group(2) or "mm"
                                cdata["gauge"] = f"{val} {unit}"
                            else:
                                cdata["gauge"] = "0.5 mm"

            # Re-serialize to clean YAML
            return yaml.dump(data, sort_keys=False, default_flow_style=False)
        except Exception:
            # Fallback to string-cleaned text
            return text

    def analyze_physical_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs automated Electrical Rule Checks (ERC) and Fluidic Rule Checks (FRC).
        """
        cables = data.get("cables", {})
        connectors = data.get("connectors", {})

        erc_results: List[Dict[str, Any]] = []
        frc_results: List[Dict[str, Any]] = []

        total_wire_length_m = 0.0
        total_tube_length_m = 0.0
        total_dead_volume_ul = 0.0
        max_pressure_drop_psi = 0.0

        is_fluidic_harness = any(
            any(kw in str(k).lower() or kw in str(v).lower() for kw in ["tube", "pump", "fluid", "cis", "trans", "luer", "falcon", "waste", "cartridge"])
            for k, v in {**cables, **connectors}.items()
        )

        # Buffer dynamic viscosity (Pa*s) at 22 C
        mu_buffer = 1.02e-3
        nominal_flow_rate_ul_min = 10.0 # 10 uL/min
        q_m3_s = (nominal_flow_rate_ul_min * 1.0e-9) / 60.0

        if isinstance(cables, dict):
            for cname, cdata in cables.items():
                if not isinstance(cdata, dict):
                    continue

                length_m = parse_length_m(cdata.get("length", 0.1))
                gauge_str = str(cdata.get("gauge", "")).strip()
                c_type = str(cdata.get("type", "")).lower()
                c_notes = str(cdata.get("notes", "")).lower()
                is_tube = any(kw in cname.lower() or kw in c_type or kw in c_notes for kw in ["tube", "pipe", "line", "hose", "fluid"])

                if is_tube:
                    total_tube_length_m += length_m

                    # Determine inner diameter in mm
                    id_mm = 0.508 # default 0.020" (~0.5 mm)
                    if "0.020" in gauge_str or "0.020" in c_type:
                        id_mm = 0.508
                    elif "0.030" in gauge_str or "0.030" in c_type:
                        id_mm = 0.762
                    elif "0.51" in gauge_str or "0.51" in c_type:
                        id_mm = 0.510
                    elif "1/32" in gauge_str or "1/32" in c_type:
                        id_mm = 0.794
                    elif "0.170" in gauge_str or "0.170" in c_type:
                        id_mm = 4.318
                    else:
                        m = re.search(r'([0-9.]+)\s*mm', gauge_str)
                        if m:
                            id_mm = float(m.group(1))

                    # Dead volume (uL): V = L * pi * (d/2)^2
                    d_m = id_mm * 1.0e-3
                    vol_m3 = length_m * math.pi * ((d_m / 2.0) ** 2)
                    vol_ul = vol_m3 * 1.0e9
                    total_dead_volume_ul += vol_ul

                    # Poiseuille laminar pressure drop (Pa): Delta P = (128 * mu * L * Q) / (pi * d^4)
                    dp_pa = (128.0 * mu_buffer * length_m * q_m3_s) / (math.pi * (d_m ** 4))
                    dp_psi = dp_pa * 0.000145038
                    max_pressure_drop_psi = max(max_pressure_drop_psi, dp_psi)

                    # Transit time (delay) in seconds: tau = V / Q
                    transit_sec = (vol_ul / nominal_flow_rate_ul_min) * 60.0

                    frc_results.append({
                        "tube": cname,
                        "type": cdata.get("type", "Tubing Line"),
                        "length_m": round(length_m, 3),
                        "id_mm": round(id_mm, 3),
                        "dead_volume_ul": round(vol_ul, 2),
                        "transit_time_s": round(transit_sec, 1),
                        "pressure_drop_psi": round(dp_psi, 4),
                        "status": "PASS" if dp_psi < 0.25 else "WARN_RESTRICTION"
                    })
                else:
                    total_wire_length_m += length_m

                    # Extract AWG number
                    awg_match = re.search(r'([0-9]{2})\s*AWG', gauge_str, re.IGNORECASE)
                    awg = int(awg_match.group(1)) if awg_match else 28
                    awg_info = AWG_DATA.get(awg, AWG_DATA[28])

                    r_wire = 2.0 * length_m * awg_info["r_per_m"] # round-trip loop resistance
                    shielded = bool(cdata.get("shield", False))

                    # Check analog noise isolation
                    is_analog = any(kw in cname.lower() or kw in c_notes for kw in ["analog", "pogo", "tia", "signal", "sensor"])
                    needs_shield = is_analog and not shielded

                    erc_results.append({
                        "wire": cname,
                        "awg": awg,
                        "length_m": round(length_m, 3),
                        "loop_resistance_ohms": round(r_wire, 4),
                        "ampacity_a": awg_info["ampacity_a"],
                        "shielded": shielded,
                        "status": "WARN_UNSHIELDED_ANALOG" if needs_shield else "PASS"
                    })

        return {
            "is_fluidic": is_fluidic_harness,
            "total_wire_length_m": round(total_wire_length_m, 3),
            "total_tube_length_m": round(total_tube_length_m, 3),
            "total_dead_volume_ul": round(total_dead_volume_ul, 2),
            "max_pressure_drop_psi": round(max_pressure_drop_psi, 4),
            "flow_delay_s": round((total_dead_volume_ul / nominal_flow_rate_ul_min) * 60.0, 1) if nominal_flow_rate_ul_min > 0 else 0,
            "erc_checks": erc_results,
            "frc_checks": frc_results,
            "overall_status": "OPTIMAL" if all(e.get("status") == "PASS" for e in erc_results + frc_results) else "ATTENTION"
        }

    def render_harness(self, raw_yaml: str) -> Dict[str, Any]:
        """
        Compiles a WireViz YAML string into SVG, structured BOM, and physical analytics.
        """
        if not WIREVIZ_AVAILABLE:
            return {
                "success": False,
                "error": "WireViz package not available in environment."
            }

        cleaned_yaml = self.preprocess_yaml(raw_yaml)

        try:
            raw_data = yaml.safe_load(cleaned_yaml) or {}
            svg_data, harness = wv_parse(cleaned_yaml, return_types=('svg', 'harness'))

            # Extract structured Bill of Materials
            bom_raw = harness.bom() if hasattr(harness, 'bom') else []
            bom_formatted = []
            for item in bom_raw:
                des_list = item.get("designators", [])
                des_str = ", ".join(str(d) for d in des_list) if isinstance(des_list, list) else str(des_list)
                bom_formatted.append({
                    "id": item.get("id"),
                    "description": item.get("description", "Component"),
                    "designators": des_str,
                    "qty": item.get("qty", 1),
                    "unit": item.get("unit", "ea"),
                    "manufacturer": item.get("manufacturer") or "-",
                    "mpn": item.get("mpn") or "-"
                })

            # Physical Rule Checks
            rules = self.analyze_physical_rules(raw_data)

            metadata = raw_data.get("metadata", {})
            title = metadata.get("title", "Wiring / Tubing Harness")

            return {
                "success": True,
                "svg": svg_data,
                "bom": bom_formatted,
                "rules": rules,
                "metadata": metadata,
                "title": title
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cleaned_yaml": cleaned_yaml
            }

    def get_project_harness(self, project_id: str, harness_type: str = "electrical") -> str:
        """
        Retrieves project YAML harness from disk or built-in templates.
        """
        # Determine candidate directory
        if project_id == "daemon-pore":
            base_dir = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore\wiring"
            fn = "reader_wiring.yaml" if harness_type == "electrical" else "reader_tubing.yaml"
        else:
            base_dir = r"c:\Users\Loki-VR\Documents\projects\Corkscrew-Filter\wiring"
            fn = "filter_interconnect.yaml"

        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"[WireVizEngine] Error reading {p}: {e}")

        # Fallback default template
        return self.get_default_template(harness_type)

    def save_project_harness(self, project_id: str, harness_type: str, yaml_content: str) -> bool:
        """
        Saves updated YAML harness back to the project wiring directory.
        """
        if project_id == "daemon-pore":
            base_dir = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore\wiring"
            fn = "reader_wiring.yaml" if harness_type == "electrical" else "reader_tubing.yaml"
        else:
            base_dir = r"c:\Users\Loki-VR\Documents\projects\Corkscrew-Filter\wiring"
            fn = "filter_interconnect.yaml"

        try:
            os.makedirs(base_dir, exist_ok=True)
            p = os.path.join(base_dir, fn)
            with open(p, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            return True
        except Exception as e:
            print(f"[WireVizEngine] Save error: {e}")
            return False

    def export_harness(self, yaml_content: str, fmt: str = "svg") -> Optional[Tuple[bytes, str, str]]:
        """
        Exports harness into downloadable binary or text format (SVG, HTML, CSV, PNG).
        Returns (data_bytes, filename, mime_type).
        """
        cleaned = self.preprocess_yaml(yaml_content)
        fmt = fmt.lower()

        try:
            if fmt == "svg":
                svg_str = wv_parse(cleaned, return_types="svg")
                return svg_str.encode("utf-8"), "harness_diagram.svg", "image/svg+xml"

            elif fmt == "csv":
                _, harness = wv_parse(cleaned, return_types=('svg', 'harness'))
                bom_items = harness.bom() if hasattr(harness, 'bom') else []
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["Line", "Designators", "Description", "Qty", "Unit", "Manufacturer", "MPN"])
                for idx, item in enumerate(bom_items, 1):
                    des_list = item.get("designators", [])
                    des_str = ", ".join(str(d) for d in des_list) if isinstance(des_list, list) else str(des_list)
                    writer.writerow([
                        idx,
                        des_str,
                        item.get("description", "Component"),
                        item.get("qty", 1),
                        item.get("unit", "ea"),
                        item.get("manufacturer") or "-",
                        item.get("mpn") or "-"
                    ])
                return out.getvalue().encode("utf-8"), "harness_bom.csv", "text/csv"

            elif fmt == "html":
                with tempfile.TemporaryDirectory() as td:
                    wv_parse(cleaned, output_formats=("html", "svg"), output_dir=td, output_name="doc")
                    p = os.path.join(td, "doc.html")
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            return f.read(), "harness_documentation.html", "text/html"

            elif fmt == "png":
                png_bytes = wv_parse(cleaned, return_types="png")
                return png_bytes, "harness_diagram.png", "image/png"

        except Exception as e:
            print(f"[WireVizEngine] Export error: {e}")
            return None

        return None

    def get_default_template(self, harness_type: str) -> str:
        if harness_type == "fluidic":
            return """metadata:
  title: Microfluidic Tubing and Fittings Harness
  description: Microbore fluidic interconnect between buffer reservoir, pump, and flowcell cartridge
  author: OpenAuto-CFD
  version: 1.0.0

options:
  fontname: Arial
  bgcolor: "#06090f"
  bgcolor_connector: "#1e293b"
  bgcolor_cable: "#0f172a"
  color_mode: SHORT

connectors:
  RES_BUFFER:
    type: Buffer Reservoir (15 mL Falcon)
    notes: 1.0 M KCl buffer supply
    pinlabels: [LIQUID_DRAW, AIR_VENT]

  PUMP_PERI:
    type: Peristaltic Pump Head
    subtype: inline
    manufacturer: Takasago
    mpn: RP-Q1.2
    pinlabels: [SUCTION_IN, DISCHARGE_OUT]

  CARTRIDGE_CIS:
    type: Flowcell Top Plate Ports (1/4-28 Flat-Bottom)
    subtype: female
    pinlabels: [PORT_IN_210, PORT_OUT_30]

  RES_WASTE:
    type: Effluent Collection Vessel
    pinlabels: [WASTE_IN]

cables:
  TUBE_PUMP:
    category: bundle
    type: PharMed BPT (0.51mm ID)
    gauge: 0.51 mm
    length: 0.12 m
    colors: [YE]

  TUBE_FEED:
    category: bundle
    type: FEP Microbore (1/16 OD x 0.020 ID)
    gauge: 0.5 mm
    length: 0.22 m
    colors: [TQ]

  TUBE_DRAIN:
    category: bundle
    type: Silicone Waste Line (1/16 OD x 1/32 ID)
    gauge: 0.8 mm
    length: 0.25 m
    colors: [BU]

connections:
  -
    - RES_BUFFER: 1
    - TUBE_PUMP: 1
    - PUMP_PERI: 1
  -
    - PUMP_PERI: 2
    - TUBE_FEED: 1
    - CARTRIDGE_CIS: 1
  -
    - CARTRIDGE_CIS: 2
    - TUBE_DRAIN: 1
    - RES_WASTE: 1
"""
        else:
            return """metadata:
  title: Electrical Wiring Harness
  description: Internal DC power and signal interconnect
  author: OpenAuto-CFD
  version: 1.0.0

options:
  fontname: Arial
  bgcolor: "#0b0f1a"
  bgcolor_connector: "#1e293b"
  bgcolor_cable: "#0f172a"
  color_mode: SHORT

connectors:
  J_PWR_IN:
    type: DC Power Jack / USB-C
    pinlabels: [VBUS, GND]

  J_MAIN_BOARD:
    type: Board Power Header (2-Pin)
    pinlabels: [VIN, GND]

cables:
  W_MAIN_PWR:
    gauge: 24 AWG
    length: 0.15 m
    colors: [RD, BK]
    notes: 2-conductor power cable

connections:
  -
    - J_PWR_IN: [1, 2]
    - W_MAIN_PWR: [1, 2]
    - J_MAIN_BOARD: [1, 2]
"""


# Singleton
_wireviz_engine: Optional[WireVizEngine] = None

def get_wireviz_engine() -> WireVizEngine:
    global _wireviz_engine
    if _wireviz_engine is None:
        _wireviz_engine = WireVizEngine()
    return _wireviz_engine
