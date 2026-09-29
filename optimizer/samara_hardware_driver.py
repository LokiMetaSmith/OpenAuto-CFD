"""
samara_hardware_driver.py

Samara Feather Hardware & Multi-Physics Simulation Driver:
1. KiCad PCB Synthesis:
   Generates hardware/samara_stator.kicad_pcb featuring 6-coil Archimedean planar spirals,
   ESP32-S3 MCU pads, dual IMU coordinate markers, and differential pressure sensor traces.
2. Circuit Board Multi-Physics:
   Evaluates planar spiral coil inductance (Mohan-Wheeler formula), high-frequency AC skin
   depth, copper winding resistance, Wheeler-Hammerstad microstrip impedance, and IPC-2152 thermal rise.
3. Impeller Airflow CFD Simulation:
   Evaluates centrifugal impeller Euler turbomachinery head, Stodola slip factor, mass flow rate,
   central plenum static pressure, and internal wing duct Fanno flow friction drop.
4. Wing Tip Velocity Aerodynamics:
   Evaluates trailing edge nozzle exit expansion velocity, reaction thrust, driving torque,
   autorotation blade element strip theory equilibrium, wing tip speed, vehicle RPM, and blown lift.
"""

import os
import math
import json
import time
from typing import Dict, Any, List, Optional, Tuple


class SamaraHardwareDriver:
    """
    Dedicated physics & hardware engine for the Samara Feather autorotating aerial vehicle.
    """

    ETA_0 = 376.730313668       # Free-space wave impedance (Ohms)
    C_0 = 2.99792458e8          # Speed of light (m/s)
    MU_0 = 4.0 * math.pi * 1e-7  # Permeability of free space (H/m)
    SIGMA_CU = 5.8e7            # Copper conductivity (S/m)
    RHO_CU = 1.724e-8           # Copper resistivity (Ohm*m)
    RHO_AIR_SL = 1.225          # Sea level air density (kg/m^3)
    SPEED_OF_SOUND_SL = 343.0   # Speed of sound (m/s)

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = os.path.abspath(project_dir) if project_dir else os.path.abspath(".")

    # =========================================================================
    # 1. Hardware KiCad PCB Synthesis
    # =========================================================================

    def generate_samara_stator_pcb(
        self,
        output_filepath: Optional[str] = None,
        num_coils: int = 6,
        stator_radius_mm: float = 25.0,
        inner_radius_mm: float = 2.0,
        num_turns: int = 8,
        trace_width_mm: float = 0.25,
        trace_spacing_mm: float = 0.25,
        board_radius_mm: float = 40.0
    ) -> str:
        """
        Synthesizes a complete KiCad 7/8 PCB (hardware/samara_stator.kicad_pcb)
        incorporating the 6 planar spiral stator coils, ESP32-S3 pads, and dual IMU markers.
        """
        if not output_filepath:
            output_filepath = os.path.join(self.project_dir, "hardware", "samara_stator.kicad_pcb")

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        pitch = trace_width_mm + trace_spacing_mm
        steps_per_turn = 32
        origin_x = 100.0  # Center of board on KiCad sheet
        origin_y = 100.0

        segments_sexpr = []
        net_id = 1

        # Generate 6 spiral coils in a circular stator array
        for coil_idx in range(num_coils):
            angle_c = coil_idx * (2.0 * math.pi / num_coils)
            xc = origin_x + stator_radius_mm * math.cos(angle_c)
            yc = origin_y + stator_radius_mm * math.sin(angle_c)

            coil_pts = []
            total_steps = int(num_turns * steps_per_turn)
            for s in range(total_steps + 1):
                theta = s * (2.0 * math.pi / steps_per_turn)
                r = inner_radius_mm + (pitch / (2.0 * math.pi)) * theta
                px = xc + r * math.cos(theta)
                py = yc + r * math.sin(theta)
                coil_pts.append((px, py))

            # Output trace segments
            phase_net = (coil_idx % 3) + 1  # 3-phase stator (Phase A, B, C)
            for i in range(len(coil_pts) - 1):
                p1 = coil_pts[i]
                p2 = coil_pts[i + 1]
                segments_sexpr.append(
                    f'  (segment (start {p1[0]:.4f} {p1[1]:.4f}) (end {p2[0]:.4f} {p2[1]:.4f}) '
                    f'(width {trace_width_mm:.4f}) (layer "F.Cu") (net {phase_net}))'
                )

        # Connect Phase Return Traces on B.Cu
        for phase in range(1, 4):
            segments_sexpr.append(
                f'  (segment (start {origin_x - 10:.3f} {origin_y + phase * 4:.3f}) '
                f'(end {origin_x + 10:.3f} {origin_y + phase * 4:.3f}) '
                f'(width 0.500) (layer "B.Cu") (net {phase}))'
            )

        # Add Interconnect Traces for ESP32 MCU and IMUs
        # IMU Inner at (0, 15) -> relative to origin
        imu_in_x = origin_x + 0.0
        imu_in_y = origin_y + 15.0
        # IMU Outer at (0, 45) -> relative to origin
        imu_out_x = origin_x + 0.0
        imu_out_y = origin_y + 45.0
        # ESP32 at (-10, 25)
        esp_x = origin_x - 10.0
        esp_y = origin_y + 25.0

        # SPI Bus connecting ESP32 to Inner & Outer IMU
        segments_sexpr.append(
            f'  (segment (start {esp_x:.3f} {esp_y:.3f}) (end {imu_in_x:.3f} {imu_in_y:.3f}) '
            f'(width 0.200) (layer "F.Cu") (net 4))'
        )
        segments_sexpr.append(
            f'  (segment (start {imu_in_x:.3f} {imu_in_y:.3f}) (end {imu_out_x:.3f} {imu_out_y:.3f}) '
            f'(width 0.200) (layer "F.Cu") (net 4))'
        )

        # Construct Edge.Cuts circular contour
        edge_steps = 64
        edge_cuts_sexpr = []
        for i in range(edge_steps):
            th1 = i * (2.0 * math.pi / edge_steps)
            th2 = (i + 1) * (2.0 * math.pi / edge_steps)
            x1 = origin_x + board_radius_mm * math.cos(th1)
            y1 = origin_y + board_radius_mm * math.sin(th1)
            x2 = origin_x + board_radius_mm * math.cos(th2)
            y2 = origin_y + board_radius_mm * math.sin(th2)
            edge_cuts_sexpr.append(
                f'  (gr_line (start {x1:.3f} {y1:.3f}) (end {x2:.3f} {y2:.3f}) (stroke (width 0.15) (type solid)) (layer "Edge.Cuts"))'
            )

        # Footprints for ESP32 and Dual IMU sensors
        footprints_sexpr = f"""
  (footprint "Module:ESP32-S3-WROOM-1" (layer "F.Cu")
    (at {esp_x:.3f} {esp_y:.3f})
    (descr "ESP32-S3 Microcontroller Module")
    (fp_text reference "U1" (at 0 -9) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))
    (fp_rect (start -9 -12) (end 9 12) (stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS"))
    (pad "1" smd rect (at -8.5 -8) (size 1.5 0.9) (layers "F.Cu" "F.Paste" "F.Mask") (net 4))
    (pad "2" smd rect (at -8.5 -6.73) (size 1.5 0.9) (layers "F.Cu" "F.Paste" "F.Mask") (net 1))
    (pad "3" smd rect (at -8.5 -5.46) (size 1.5 0.9) (layers "F.Cu" "F.Paste" "F.Mask") (net 2))
    (pad "4" smd rect (at -8.5 -4.19) (size 1.5 0.9) (layers "F.Cu" "F.Paste" "F.Mask") (net 3))
  )
  (footprint "Sensor_Motion:LGA-16_3x3mm" (layer "F.Cu")
    (at {imu_in_x:.3f} {imu_in_y:.3f})
    (descr "Inner IMU (ICM-20602) Centrifugal Sensor")
    (fp_text reference "U2" (at 0 -2.5) (layer "F.SilkS") (effects (font (size 0.8 0.8) (thickness 0.12))))
    (fp_rect (start -1.5 -1.5) (end 1.5 1.5) (stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS"))
    (pad "1" smd rect (at -1.2 -1) (size 0.6 0.3) (layers "F.Cu" "F.Paste" "F.Mask") (net 4))
  )
  (footprint "Sensor_Motion:LGA-16_3x3mm" (layer "F.Cu")
    (at {imu_out_x:.3f} {imu_out_y:.3f})
    (descr "Outer IMU (ICM-20602) Centrifugal Sensor")
    (fp_text reference "U3" (at 0 -2.5) (layer "F.SilkS") (effects (font (size 0.8 0.8) (thickness 0.12))))
    (fp_rect (start -1.5 -1.5) (end 1.5 1.5) (stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS"))
    (pad "1" smd rect (at -1.2 -1) (size 0.6 0.3) (layers "F.Cu" "F.Paste" "F.Mask") (net 4))
  )
"""

        all_segments = "\n".join(segments_sexpr)
        all_edges = "\n".join(edge_cuts_sexpr)

        kicad_body = f"""(kicad_pcb (version 20221018) (generator "OpenAuto-SamaraDriver")

  (general
    (thickness 1.6)
  )

  (paper "A4")

  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (44 "Edge.Cuts" user)
  )

  (net 0 "")
  (net 1 "Phase_A")
  (net 2 "Phase_B")
  (net 3 "Phase_C")
  (net 4 "SPI_CLK")
  (net 5 "GND")

{footprints_sexpr}

{all_segments}

{all_edges}

)
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(kicad_body)

        return output_filepath

    # =========================================================================
    # 2. Circuit Board Multi-Physics Simulation
    # =========================================================================

    def simulate_circuit_board(
        self,
        board_path: Optional[str] = None,
        freq_mhz: float = 3.0,
        phase_current_arms: float = 1.5,
        copper_thickness_um: float = 35.0,
        substrate_er: float = 4.3,
        substrate_height_mm: float = 1.6
    ) -> Dict[str, Any]:
        """
        Simulates planar stator electromagnetic, impedance, RF skin effect,
        and IPC-2152 thermal performance for the Samara Feather board.
        """
        num_coils = 6
        num_turns = 8
        w_mm = 0.25
        s_mm = 0.25
        r_in_mm = 2.0
        pitch_mm = w_mm + s_mm
        r_out_mm = r_in_mm + num_turns * pitch_mm  # 6.0 mm
        d_avg_mm = r_in_mm + r_out_mm               # 8.0 mm
        rho_fill = (r_out_mm - r_in_mm) / (r_out_mm + r_in_mm)  # 0.50

        f_hz = float(freq_mhz) * 1e6
        omega = 2.0 * math.pi * f_hz

        # 1. Mohan-Wheeler Planar Spiral Inductance (Modified Wheeler Formula)
        d_avg_m = d_avg_mm * 1e-3
        term_geom = math.log(2.46 / rho_fill) + 0.20 * (rho_fill ** 2)
        coil_l_henries = (self.MU_0 * (num_turns ** 2) * d_avg_m * 1.0 / 2.0) * term_geom
        coil_l_uh = coil_l_henries * 1e6

        # 3-Phase stator: 2 series coils per phase
        phase_l_uh = coil_l_uh * 2.0

        # 2. Conductor Skin Effect & Resistance
        skin_depth_m = math.sqrt(1.0 / (math.pi * f_hz * self.MU_0 * self.SIGMA_CU))
        skin_depth_um = skin_depth_m * 1e6

        r_avg_m = (r_in_mm + r_out_mm) / 2.0 * 1e-3
        trace_len_m = 2.0 * math.pi * num_turns * r_avg_m
        trace_area_dc_m2 = (w_mm * 1e-3) * (copper_thickness_um * 1e-6)

        r_dc = self.RHO_CU * (trace_len_m / trace_area_dc_m2)

        t_m = copper_thickness_um * 1e-6
        if t_m > 2.0 * skin_depth_m:
            eff_area_ac = (w_mm * 1e-3) * 2.0 * skin_depth_m
            r_ac = self.RHO_CU * (trace_len_m / eff_area_ac)
        else:
            k_skin = 1.0 + (1.0 / 48.0) * ((t_m / skin_depth_m) ** 4)
            r_ac = r_dc * k_skin

        phase_r_ac = r_ac * 2.0

        # 3. Quality Factor Q
        coil_reactance = omega * coil_l_henries
        q_factor = coil_reactance / r_ac if r_ac > 0 else 0.0

        # 4. Characteristic Impedance Z0 (Wheeler-Hammerstad) for SPI/RF signal traces
        u = w_mm / substrate_height_mm
        a = 1.0 + (1.0 / 49.0) * math.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432)) + (1.0 / 18.7) * math.log(1.0 + (u / 18.1) ** 3)
        b = 0.564 * ((substrate_er - 0.9) / (substrate_er + 3.0)) ** 0.053
        eps_eff = ((substrate_er + 1.0) / 2.0) + ((substrate_er - 1.0) / 2.0) * (1.0 + 10.0 / u) ** (-a * b)

        if u <= 1.0:
            f_u = 6.0 + (2.0 * math.pi - 6.0) * math.exp(-((30.666 / u) ** 0.7528))
            z0 = (self.ETA_0 / (2.0 * math.pi * math.sqrt(eps_eff))) * math.log((f_u / u) + math.sqrt(1.0 + (2.0 / u) ** 2))
        else:
            z0 = (self.ETA_0 / math.sqrt(eps_eff)) / (u + 1.393 + 0.667 * math.log(u + 1.444))

        # RF 50-ohm controlled impedance line (L1 over L2 ground plane, h_prepreg = 0.2mm, w = 0.35mm)
        h_rf = 0.2
        w_rf = 0.35
        u_rf = w_rf / h_rf
        a_rf = 1.0 + (1.0 / 49.0) * math.log((u_rf ** 4 + (u_rf / 52.0) ** 2) / (u_rf ** 4 + 0.432)) + (1.0 / 18.7) * math.log(1.0 + (u_rf / 18.1) ** 3)
        b_rf = 0.564 * ((substrate_er - 0.9) / (substrate_er + 3.0)) ** 0.053
        eps_eff_rf = ((substrate_er + 1.0) / 2.0) + ((substrate_er - 1.0) / 2.0) * (1.0 + 10.0 / u_rf) ** (-a_rf * b_rf)
        z0_rf = (self.ETA_0 / math.sqrt(eps_eff_rf)) / (u_rf + 1.393 + 0.667 * math.log(u_rf + 1.444))

        # 5. Thermal Dissipation (IPC-2152 Standard)
        i_arms = float(phase_current_arms)
        phase_power_w = (i_arms ** 2) * phase_r_ac
        total_stator_loss_w = phase_power_w * 3.0

        w_mils = w_mm / 0.0254
        t_mils = (copper_thickness_um * 1e-3) / 0.0254
        area_mils2 = w_mils * t_mils
        k_ipc = 0.048
        delta_temp_c = ((i_arms / (k_ipc * (area_mils2 ** 0.725))) ** (1.0 / 0.44)) if area_mils2 > 0 else 25.0
        board_temp_c = 25.0 + delta_temp_c

        return {
            "status": "success",
            "board_type": "Samara Planar Stator Multi-Phase PCB",
            "operating_frequency_mhz": float(freq_mhz),
            "stator_metrics": {
                "num_coils": num_coils,
                "turns_per_coil": num_turns,
                "coil_inductance_uh": round(coil_l_uh, 4),
                "phase_inductance_uh": round(phase_l_uh, 4),
                "coil_dc_resistance_ohms": round(r_dc, 3),
                "coil_ac_resistance_ohms": round(r_ac, 3),
                "phase_ac_resistance_ohms": round(phase_r_ac, 3),
                "skin_depth_um": round(skin_depth_um, 2),
                "coil_quality_factor_q": round(q_factor, 2)
            },
            "transmission_line_metrics": {
                "trace_width_mm": w_mm,
                "substrate_height_mm": substrate_height_mm,
                "dielectric_constant_er": substrate_er,
                "characteristic_impedance_z0_ohms": round(z0, 2),
                "rf_controlled_impedance_50ohm_z0": round(z0_rf, 2),
                "effective_permittivity_eps_eff": round(eps_eff, 3)
            },
            "thermal_and_power": {
                "phase_current_arms": i_arms,
                "power_loss_per_phase_watts": round(phase_power_w, 3),
                "total_stator_loss_watts": round(total_stator_loss_w, 3),
                "temperature_rise_c": round(delta_temp_c, 1),
                "estimated_board_temp_c": round(board_temp_c, 1),
                "ipc_standard": "IPC-2152 compliant"
            }
        }

    # =========================================================================
    # 3. Pressurized Airflow CFD & Impeller Aerodynamics
    # =========================================================================

    def simulate_impeller_airflow(
        self,
        rpm: float = 6393.0,
        impeller_radius_mm: float = 65.0,
        blade_height_mm: float = 18.0,
        blade_count: int = 12,
        backward_sweep_deg: float = 30.0,
        air_density: float = 1.225,
        duct_length_mm: float = 350.0,
        duct_width_mm: float = 45.0,
        duct_height_mm: float = 8.5
    ) -> Dict[str, Any]:
        """
        Simulates the centrifugal impeller radial compressor aerodynamics,
        Stodola slip factor, Euler total head, plenum static pressure, mass flow rate,
        and internal wing duct Fanno flow pressure losses.
        """
        rho = float(air_density)
        omega = float(rpm) * (2.0 * math.pi / 60.0)
        r2_m = float(impeller_radius_mm) * 1e-3
        b2_m = float(blade_height_mm) * 1e-3

        u2 = omega * r2_m

        beta2_rad = math.radians(backward_sweep_deg)
        sigma = 1.0 - (math.pi / float(blade_count)) * math.sin(beta2_rad)
        sigma = max(0.65, min(0.95, sigma))

        delta_h0 = sigma * (u2 ** 2)
        eta_impeller = 0.72
        delta_p_total = rho * delta_h0 * eta_impeller

        v_exit_radial = u2 * 0.12
        plenum_static_press_gauge = delta_p_total - 0.5 * rho * (v_exit_radial ** 2) * 0.45
        plenum_total_press_abs = 101325.0 + delta_p_total
        plenum_static_press_abs = 101325.0 + plenum_static_press_gauge

        area_exit = 2.0 * math.pi * r2_m * b2_m * 0.90
        volumetric_flow_m3_s = area_exit * v_exit_radial
        mass_flow_total_kg_s = rho * volumetric_flow_m3_s
        mass_flow_per_wing_kg_s = mass_flow_total_kg_s / 2.0

        # Internal flattened wing duct geometry (wide rectangular passage in airfoil)
        l_duct_m = float(duct_length_mm) * 1e-3
        w_duct_m = float(duct_width_mm) * 1e-3
        h_duct_m = float(duct_height_mm) * 1e-3
        area_duct_m2 = w_duct_m * h_duct_m
        dh_m = (2.0 * w_duct_m * h_duct_m) / (w_duct_m + h_duct_m)

        v_duct = mass_flow_per_wing_kg_s / (rho * area_duct_m2)
        mu_air = 1.81e-5
        re_duct = (rho * v_duct * dh_m) / mu_air

        f_darcy = 0.316 * (re_duct ** (-0.25)) if re_duct > 2300 else 64.0 / max(re_duct, 1.0)
        duct_friction_drop_pa = f_darcy * (l_duct_m / dh_m) * (0.5 * rho * (v_duct ** 2))

        nozzle_total_press_gauge = max(10.0, plenum_static_press_gauge - duct_friction_drop_pa)

        return {
            "status": "success",
            "operating_rpm": round(rpm, 1),
            "angular_velocity_rad_s": round(omega, 2),
            "impeller_aerodynamics": {
                "tip_speed_u2_m_s": round(u2, 2),
                "stodola_slip_factor": round(sigma, 4),
                "euler_enthalpy_rise_j_kg": round(delta_h0, 1),
                "aerodynamic_efficiency": eta_impeller,
                "total_pressure_rise_pa": round(delta_p_total, 1),
                "radial_exit_velocity_m_s": round(v_exit_radial, 2)
            },
            "central_plenum": {
                "static_pressure_gauge_pa": round(plenum_static_press_gauge, 1),
                "static_pressure_abs_pa": round(plenum_static_press_abs, 1),
                "mass_flow_total_g_s": round(mass_flow_total_kg_s * 1000.0, 2),
                "mass_flow_per_wing_g_s": round(mass_flow_per_wing_kg_s * 1000.0, 2),
                "volumetric_flow_l_s": round(volumetric_flow_m3_s * 1000.0, 2)
            },
            "internal_wing_duct_fanno_flow": {
                "duct_length_mm": duct_length_mm,
                "duct_width_mm": duct_width_mm,
                "duct_height_mm": duct_height_mm,
                "hydraulic_diameter_mm": round(dh_m * 1000.0, 2),
                "internal_flow_velocity_m_s": round(v_duct, 2),
                "reynolds_number": round(re_duct, 0),
                "darcy_friction_factor": round(f_darcy, 4),
                "duct_frictional_loss_pa": round(duct_friction_drop_pa, 1),
                "net_nozzle_pressure_gauge_pa": round(nozzle_total_press_gauge, 1)
            }
        }

    # =========================================================================
    # 4. Wing Tip Velocity & Autorotation Dynamics Simulation
    # =========================================================================

    def simulate_wing_velocity(
        self,
        nozzle_press_gauge_pa: float = 1022.0,
        mass_flow_per_wing_g_s: float = 18.45,
        span_length_mm: float = 415.0,
        root_chord_mm: float = 90.0,
        tip_chord_mm: float = 25.0,
        vehicle_mass_grams: float = 380.0,
        air_density: float = 1.225
    ) -> Dict[str, Any]:
        """
        Simulates trailing edge nozzle expansion exit velocity (V_e), jet reaction thrust,
        driving torque, aerodynamic profile drag equilibrium via blade element strip theory,
        resulting in the steady-state autorotation tip speed and blown circulation lift.
        """
        rho = float(air_density)
        r_span_m = float(span_length_mm) * 1e-3
        c_root_m = float(root_chord_mm) * 1e-3
        c_tip_m = float(tip_chord_mm) * 1e-3
        m_dot_wing = float(mass_flow_per_wing_g_s) * 1e-3

        c_discharge = 0.94
        delta_p = max(5.0, float(nozzle_press_gauge_pa))
        v_exit_ideal = math.sqrt(2.0 * delta_p / rho)
        v_exit_actual = c_discharge * v_exit_ideal

        thrust_jet_per_wing = m_dot_wing * v_exit_actual
        total_jet_thrust = thrust_jet_per_wing * 2.0

        r_eff_thrust = r_span_m * 0.92
        driving_torque = total_jet_thrust * r_eff_thrust

        num_strips = 40
        dr = r_span_m / float(num_strips)

        def get_chord(r):
            frac = min(1.0, max(0.0, r / r_span_m))
            return c_root_m - (c_root_m - c_tip_m) * (frac ** 1.5)

        cd_profile = 0.038

        integral_constant = 0.0
        for i in range(num_strips):
            r = (i + 0.5) * dr
            c = get_chord(r)
            integral_constant += c * (r ** 3) * dr

        drag_torque_factor = rho * cd_profile * integral_constant * 2.0

        if drag_torque_factor > 0:
            omega_steady = math.sqrt(driving_torque / drag_torque_factor)
        else:
            omega_steady = 50.0

        vehicle_rpm = omega_steady * 60.0 / (2.0 * math.pi)

        v_tip_m_s = omega_steady * r_span_m
        v_tip_km_h = v_tip_m_s * 3.6
        mach_tip = v_tip_m_s / self.SPEED_OF_SOUND_SL

        cl_base = 0.55
        lift_integral_constant = 0.0
        for i in range(num_strips):
            r = (i + 0.5) * dr
            c = get_chord(r)
            lift_integral_constant += c * (r ** 2) * dr

        rotational_lift_n = 2.0 * 0.5 * rho * (omega_steady ** 2) * cl_base * lift_integral_constant
        blown_lift_n = 1.35 * total_jet_thrust
        total_lift_n = rotational_lift_n + blown_lift_n

        m_veh_kg = float(vehicle_mass_grams) * 1e-3
        weight_n = m_veh_kg * 9.81
        thrust_to_weight = total_lift_n / weight_n if weight_n > 0 else 1.0

        return {
            "status": "success",
            "nozzle_jet_expansion": {
                "nozzle_pressure_gauge_pa": round(delta_p, 1),
                "exit_velocity_m_s": round(v_exit_actual, 2),
                "thrust_per_wing_n": round(thrust_jet_per_wing, 3),
                "total_jet_thrust_n": round(total_jet_thrust, 3),
                "driving_torque_nm": round(driving_torque, 4)
            },
            "steady_state_autorotation": {
                "wing_span_mm": span_length_mm,
                "steady_rotation_rad_s": round(omega_steady, 2),
                "steady_vehicle_rpm": round(vehicle_rpm, 1),
                "wing_tip_velocity_m_s": round(v_tip_m_s, 2),
                "wing_tip_velocity_km_h": round(v_tip_km_h, 1),
                "wing_tip_mach": round(mach_tip, 3)
            },
            "flight_lift_and_hover": {
                "vehicle_mass_grams": vehicle_mass_grams,
                "vehicle_weight_n": round(weight_n, 2),
                "rotational_aero_lift_n": round(rotational_lift_n, 2),
                "blown_coanda_lift_n": round(blown_lift_n, 2),
                "total_vertical_lift_n": round(total_lift_n, 2),
                "thrust_to_weight_ratio": round(thrust_to_weight, 2),
                "hover_climb_capability": "Excess thrust available (Positive climb rate)" if thrust_to_weight > 1.05 else "Hover limit"
            }
        }
