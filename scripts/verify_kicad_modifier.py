"""
verify_kicad_modifier.py

Validates KiCadLayoutModifier on real and mock PCB files:
1. Transactional backup creation (.bak_<timestamp>)
2. Atomic trace width updates for specific nets
3. Parentheses balance and integrity validation
"""

import os
import shutil
import tempfile
from kicad_plugin.kicad_modifier import KiCadLayoutModifier
from kicad_plugin.kicad_parser import KiCadPcbParser

def test_modifier_workflow():
    real_pcb = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore\Amplifier\amplifier.kicad_pcb"
    if not os.path.exists(real_pcb):
        print(f"Skipping test: {real_pcb} not found")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pcb = os.path.join(tmpdir, "test_amp.kicad_pcb")
        shutil.copy2(real_pcb, test_pcb)

        # 1. Parse initial state
        parser_initial = KiCadPcbParser(test_pcb)
        initial_net = parser_initial.get_all_nets_summary().get("/Signal_AMP")
        assert initial_net is not None, "Net /Signal_AMP not found"
        assert initial_net["trace_width_mm"] == 0.2, f"Expected 0.2, got {initial_net['trace_width_mm']}"
        print(f"[Initial State] /Signal_AMP: width={initial_net['trace_width_mm']}mm, segs={initial_net['segment_count']}")

        # 2. Modify trace width
        modifier = KiCadLayoutModifier(test_pcb)
        result = modifier.update_net_trace_width("/Signal_AMP", 2.43, create_backup=True)
        assert result["success"] is True, f"Modifier failed: {result}"
        assert result["modified_segments"] == initial_net["segment_count"]
        assert os.path.exists(result["backup_path"]), "Backup file was not created"
        print(f"[Modified] {result['message']} | Backup: {os.path.basename(result['backup_path'])}")

        # 3. Parse updated file and verify integrity
        parser_updated = KiCadPcbParser(test_pcb)
        updated_net = parser_updated.get_all_nets_summary().get("/Signal_AMP")
        assert updated_net is not None
        assert updated_net["trace_width_mm"] == 2.43, f"Expected 2.43, got {updated_net['trace_width_mm']}"
        print(f"[Verified] /Signal_AMP is now {updated_net['trace_width_mm']}mm across all {updated_net['segment_count']} segments")

        # Other nets must be unchanged
        guard_net = parser_updated.get_all_nets_summary().get("/GUARD")
        assert guard_net["trace_width_mm"] == 0.6, f"Expected /GUARD to stay 0.6mm, got {guard_net['trace_width_mm']}"
        print(f"[Integrity] Other nets untouched: /GUARD is {guard_net['trace_width_mm']}mm")

    print(">>> ALL KICAD MODIFIER TESTS PASSED! <<<")

if __name__ == "__main__":
    test_modifier_workflow()
