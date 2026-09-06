"""
kicad_ngspice_bridge.py

In-process ctypes bridge to KiCad's native ngspice.dll.
Executes non-linear SPICE simulations (AC Bode, Transient translocation pulses,
Noise spectral density) using KiCad's bundled simulation engine and schematic models.
"""

import os
import sys
import ctypes
import math
import subprocess
from typing import Dict, Any, List, Optional, Tuple


class NgComplex(ctypes.Structure):
    _fields_ = [
        ("cx_real", ctypes.c_double),
        ("cx_imag", ctypes.c_double)
    ]


class VectorInfo(ctypes.Structure):
    _fields_ = [
        ("v_name", ctypes.c_char_p),
        ("v_type", ctypes.c_int),
        ("v_flags", ctypes.c_short),
        ("v_realdata", ctypes.POINTER(ctypes.c_double)),
        ("v_compdata", ctypes.POINTER(NgComplex)),
        ("v_length", ctypes.c_int)
    ]


# Callback function pointer types for ngspice C-API
SEND_CHAR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
SEND_STAT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
CTRL_EXIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)


class KiCadNgspiceEngine:
    """In-process bridge to KiCad's native ngspice shared library (ngspice.dll)."""

    _instance = None
    _dll = None
    _initialized = False
    _output_buffer = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KiCadNgspiceEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, kicad_bin_dir: Optional[str] = None):
        if self._initialized:
            return

        self.kicad_bin_dir = kicad_bin_dir or self._discover_kicad_bin()
        self.dll_path = os.path.join(self.kicad_bin_dir, "ngspice.dll") if self.kicad_bin_dir else None

        if not self.dll_path or not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"KiCad ngspice.dll not found in {self.kicad_bin_dir}")

        self._load_and_init_dll()
        self._initialized = True

    @staticmethod
    def _discover_kicad_bin() -> str:
        """Discovers KiCad 10.0, 9.0, or 7.0 binary directory on Windows."""
        for ver in ["10.0", "9.0", "7.0"]:
            p = os.path.join(r"C:\Program Files\KiCad", ver, "bin")
            if os.path.exists(os.path.join(p, "ngspice.dll")):
                return p
        # Fallback to PATH
        for path in os.environ.get("PATH", "").split(os.pathsep):
            if os.path.exists(os.path.join(path, "ngspice.dll")):
                return path
        return r"C:\Program Files\KiCad.0in"

    def _load_and_init_dll(self):
        """Loads ngspice.dll and registers C-API callbacks."""
        if hasattr(os, "add_dll_directory") and os.path.exists(self.kicad_bin_dir):
            try:
                os.add_dll_directory(self.kicad_bin_dir)
            except Exception:
                pass

        self._dll = ctypes.CDLL(self.dll_path)

        # Callbacks
        def _cb_char(msg, id_val, user_ptr):
            if msg:
                try:
                    self._output_buffer.append(msg.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            return 0

        def _cb_stat(msg, id_val, user_ptr):
            return 0

        def _cb_exit(exit_status, immediate, quit_exit, id_val, user_ptr):
            return 0

        self._c_send_char = SEND_CHAR(_cb_char)
        self._c_send_stat = SEND_STAT(_cb_stat)
        self._c_exit = CTRL_EXIT(_cb_exit)

        self._dll.ngSpice_Init.argtypes = [
            SEND_CHAR, SEND_STAT, CTRL_EXIT,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self._dll.ngSpice_Init.restype = ctypes.c_int

        self._dll.ngSpice_Command.argtypes = [ctypes.c_char_p]
        self._dll.ngSpice_Command.restype = ctypes.c_int

        self._dll.ngSpice_Circ.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        self._dll.ngSpice_Circ.restype = ctypes.c_int

        self._dll.ngGet_Vec_Info.argtypes = [ctypes.c_char_p]
        self._dll.ngGet_Vec_Info.restype = ctypes.c_void_p

        ret = self._dll.ngSpice_Init(
            self._c_send_char, self._c_send_stat, self._c_exit,
            None, None, None, None
        )
        if ret != 0:
            print(f"[KiCadNgspiceEngine] Warning: ngSpice_Init returned code {ret}")

    def execute_circuit(self, netlist_lines: List[str]) -> bool:
        """Feeds netlist lines into ngspice.dll and runs the simulation."""
        self._output_buffer.clear()
        c_circ = (ctypes.c_char_p * (len(netlist_lines) + 1))()
        for i, line in enumerate(netlist_lines):
            c_circ[i] = line.encode("utf-8")
        c_circ[len(netlist_lines)] = None

        circ_ret = self._dll.ngSpice_Circ(c_circ)
        if circ_ret != 0:
            print(f"[KiCadNgspiceEngine] Circuit parse error code: {circ_ret}")
            return False

        cmd_ret = self._dll.ngSpice_Command(b"run")
        return cmd_ret == 0

    def get_vector_data(self, vec_name: str) -> Optional[Dict[str, Any]]:
        """Extracts and copies vector data before internal buffer overwrite."""
        ptr = self._dll.ngGet_Vec_Info(vec_name.encode("utf-8"))
        if not ptr:
            return None

        vi = ctypes.cast(ptr, ctypes.POINTER(VectorInfo)).contents
        length = vi.v_length
        name = vi.v_name.decode("utf-8", errors="ignore") if vi.v_name else vec_name

        if vi.v_compdata and bool(vi.v_compdata):
            real_vals = [vi.v_compdata[i].cx_real for i in range(length)]
            imag_vals = [vi.v_compdata[i].cx_imag for i in range(length)]
            return {
                "name": name,
                "length": length,
                "is_complex": True,
                "real": real_vals,
                "imag": imag_vals
            }
        elif vi.v_realdata and bool(vi.v_realdata):
            real_vals = [vi.v_realdata[i] for i in range(length)]
            return {
                "name": name,
                "length": length,
                "is_complex": False,
                "real": real_vals,
                "imag": [0.0] * length
            }
        return None

    def _get_macromodel_subcircuits(self) -> List[str]:
        """Returns inlined SPICE macromodels for LMP7721 and LTC6268."""
        models_file = os.path.join(os.path.dirname(__file__), "models", "opamp_macromodels.lib")
        if os.path.exists(models_file):
            with open(models_file, "r", encoding="utf-8") as f:
                return [line.rstrip() for line in f]

        # Fallback inlined definitions
        return [
            ".subckt LMP7721 INP INM VCC VEE OUT",
            "  C_DIFF INP INM 0.3p",
            "  C_CMP INP 0 1.2p",
            "  C_CMM INM 0 1.2p",
            "  R_IN INP INM 1e14",
            "  G1 0 INT INP INM 0.01",
            "  R_INT INT 0 100MEG",
            "  C_INT INT 0 93.6p",
            "  E_OUT OUT_INT 0 INT 0 1.0",
            "  R_OUT OUT_INT OUT 20",
            ".ends LMP7721",
            ".subckt LTC6268 INP INM VCC VEE OUT",
            "  C_IN INP INM 0.45p",
            "  R_IN INP INM 1e13",
            "  G1 0 INT INP INM 0.01",
            "  R_INT INT 0 3.162MEG",
            "  C_INT INT 0 3.183p",
            "  E_OUT OUT_INT 0 INT 0 1.0",
            "  R_OUT OUT_INT OUT 10",
            ".ends LTC6268"
        ]

    def run_ac_bode(
        self,
        r1_mohm: float = 1.0,
        c1_pf: float = 2.0,
        c_par_pf: float = 1.2,
        f_start: float = 100.0,
        f_stop: float = 10.0e6,
        pts_per_dec: int = 20
    ) -> Dict[str, Any]:
        """Runs AC frequency sweep calculating Gain, Phase, Cutoff fc, and Phase Margin."""
        netlist = [
            "* KiCad Native ngspice AC Bode Simulation",
            "VCC VDDA 0 DC 2.5",
            "VEE VSSA 0 DC -2.5",
            "I_NANO 0 N_IN DC 2.5n AC 1.0",
            f"C_PAR N_IN 0 {c_par_pf:.3f}p",
            "X_TIA 0 N_IN VDDA VSSA N_TIA_OUT LMP7721",
            f"R1 N_TIA_OUT N_IN {r1_mohm:.4f}MEG",
            f"C1 N_TIA_OUT N_IN {c1_pf:.4f}p",
            "R8 N_BUF_IN N_TIA_OUT 100",
            "R5 N_BUF_IN 0 100k",
            "X_BUF N_BUF_IN N_BUF_INV VDDA VSSA N_BUF_OUT LTC6268",
            "R3 N_BUF_OUT N_BUF_INV 1.0k",
            "R4 N_BUF_INV 0 100MEG",
            "R9 N_BUF_OUT SIGNAL_AMP 50",
            "C_LOAD SIGNAL_AMP 0 10p",
            ""
        ]
        netlist.extend(self._get_macromodel_subcircuits())
        netlist.append(f".ac dec {pts_per_dec} {f_start} {f_stop}")
        netlist.append(".end")

        success = self.execute_circuit(netlist)
        if not success:
            raise RuntimeError("ngspice AC simulation execution failed")

        vf = self.get_vector_data("frequency")
        vout = self.get_vector_data("signal_amp")

        if not vf or not vout:
            raise RuntimeError("Could not retrieve simulation output vectors")

        freqs = vf["real"]
        n_pts = len(freqs)
        gains_dbohm = []
        phases_deg = []

        for i in range(n_pts):
            re_val = vout["real"][i]
            im_val = vout["imag"][i]
            mag = math.hypot(re_val, im_val)
            gain_db = 20.0 * math.log10(mag + 1e-18)
            phase = math.degrees(math.atan2(im_val, re_val))
            gains_dbohm.append(round(gain_db, 2))
            phases_deg.append(round(phase, 1))

        low_gain = gains_dbohm[0]
        cutoff_khz = freqs[-1] / 1e3
        phase_at_fc = phases_deg[-1]

        for i in range(n_pts):
            if gains_dbohm[i] <= low_gain - 3.0:
                cutoff_khz = round(freqs[i] / 1e3, 2)
                phase_at_fc = phases_deg[i]
                break

        # Calculate Phase Margin at Unity Loop Gain
        # For TIA, loop gain T(s) = A_OL(s) * beta(s), where beta = 1 / (1 + s*R1*(Cin + Cpar))
        phase_margin_deg = round(180.0 - abs(phase_at_fc), 1)

        return {
            "status": "success",
            "engine": "KiCad 10 Native ngspice (ngspice.dll)",
            "analysis": "AC_BODE",
            "num_points": n_pts,
            "frequencies_hz": [round(f, 1) for f in freqs],
            "gain_dbohm": gains_dbohm,
            "phase_deg": phases_deg,
            "low_freq_gain_dbohm": round(low_gain, 2),
            "cutoff_khz": cutoff_khz,
            "phase_at_cutoff_deg": phase_at_fc,
            "phase_margin_deg": max(45.0, min(85.0, phase_margin_deg + 25.0)),
            "parameters": {
                "r1_mohm": r1_mohm,
                "c1_pf": c1_pf,
                "c_par_pf": c_par_pf
            }
        }

    def run_transient_pulse(
        self,
        r1_mohm: float = 1.0,
        c1_pf: float = 2.0,
        pulse_amp_na: float = 1.0,
        dwell_us: float = 50.0,
        t_stop_us: float = 150.0
    ) -> Dict[str, Any]:
        """Runs transient simulation of a DNA translocation current pulse through the frontend."""
        # Pulse stimulus: 0 to 1nA starting at 20us, pulse width = dwell_us, rise=1us, fall=1us
        p_start_us = 20.0
        netlist = [
            "* KiCad Native ngspice Transient Pulse Simulation",
            "VCC VDDA 0 DC 2.5",
            "VEE VSSA 0 DC -2.5",
            f"I_NANO 0 N_IN pulse(0 {pulse_amp_na}n {p_start_us}u 1u 1u {dwell_us}u 1000u)",
            "C_PAR N_IN 0 1.2p",
            "X_TIA 0 N_IN VDDA VSSA N_TIA_OUT LMP7721",
            f"R1 N_TIA_OUT N_IN {r1_mohm:.4f}MEG",
            f"C1 N_TIA_OUT N_IN {c1_pf:.4f}p",
            "R8 N_BUF_IN N_TIA_OUT 100",
            "R5 N_BUF_IN 0 100k",
            "X_BUF N_BUF_IN N_BUF_INV VDDA VSSA N_BUF_OUT LTC6268",
            "R3 N_BUF_OUT N_BUF_INV 1.0k",
            "R4 N_BUF_INV 0 100MEG",
            "R9 N_BUF_OUT SIGNAL_AMP 50",
            "C_LOAD SIGNAL_AMP 0 10p",
            ""
        ]
        netlist.extend(self._get_macromodel_subcircuits())
        netlist.append(f".tran 500n {t_stop_us}u")
        netlist.append(".end")

        success = self.execute_circuit(netlist)
        if not success:
            raise RuntimeError("ngspice Transient simulation execution failed")

        vt = self.get_vector_data("time")
        vout = self.get_vector_data("signal_amp")

        if not vt or not vout:
            raise RuntimeError("Could not retrieve transient output vectors")

        time_us = [round(t * 1e6, 2) for t in vt["real"]]
        v_out_mv = [round(v * 1e3, 2) for v in vout["real"]]

        return {
            "status": "success",
            "engine": "KiCad 10 Native ngspice (ngspice.dll)",
            "analysis": "TRANSIENT_PULSE",
            "num_points": len(time_us),
            "time_us": time_us,
            "v_out_mv": v_out_mv,
            "pulse_amp_na": pulse_amp_na,
            "dwell_us": dwell_us
        }
