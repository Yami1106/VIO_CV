"""
generate_all_scenes.py
----------------------
Generates all 10 scenes with 100s duration each.
100s × 100Hz = 10,000 poses per scene
stride 10 = 1,000 images per scene = 999 sample pairs

Total: 7 train × 999 = 6,993 train samples
       1 val  × 999 =   999 val samples
       2 test × 999 = 1,998 test samples

Usage:
    python generate_all_scenes.py
"""

import subprocess
import sys

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
GEN_SCRIPT = f"{PROJECT_ROOT}/Code/Phase2/data_generation/generate_trajectory.py"

SCENES = [
    # --- TRAIN (7 scenes: mix of simple + complex) ---
    {"name": "scene_001_circle",    "traj": "circle",    "duration": 100, "scale": 3.0},
    {"name": "scene_002_oval",      "traj": "oval",      "duration": 100, "scale": 3.0},
    {"name": "scene_003_diamond",   "traj": "diamond",   "duration": 100, "scale": 3.0},
    {"name": "scene_004_figure8",   "traj": "figure8",   "duration": 100, "scale": 3.0},
    {"name": "scene_005_star",      "traj": "star",      "duration": 100, "scale": 3.0},
    {"name": "scene_006_clover",    "traj": "clover",    "duration": 100, "scale": 3.0},
    {"name": "scene_007_mouse",     "traj": "mouse",     "duration": 100, "scale": 3.0},

    # --- VAL (1 scene) ---
    {"name": "scene_008_halfmoon",  "traj": "halfmoon",  "duration": 100, "scale": 3.0},

    # --- TEST (2 scenes: complex unseen) ---
    {"name": "scene_009_sid",       "traj": "sid",       "duration": 100, "scale": 3.0},
    {"name": "scene_010_picasso",   "traj": "picasso",   "duration": 100, "scale": 3.0},
]


def main():
    print("=" * 60)
    print("  Generating all scenes (100s each, 10k poses)")
    print("=" * 60)

    for i, scene in enumerate(SCENES):
        print(f"\n[{i+1}/{len(SCENES)}] {scene['name']} ({scene['traj']}, {scene['duration']}s)")

        cmd = [
            sys.executable, GEN_SCRIPT,
            "--scene_name", scene["name"],
            "--traj_type", scene["traj"],
            "--duration", str(scene["duration"]),
            "--rate_hz", "100",
            "--scale", str(scene["scale"]),
            "--height", "6.0",
            "--z_variation", "0.3",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(f"  {result.stdout.strip()}")

    print(f"\n{'='*60}")
    print("  Done. Now render each scene:")
    print()
    for scene in SCENES:
        print(f"  blender -b /home/yami/Downloads/Group9_p4/data.blend -P render_full_sequence.py -- --scene_name {scene['name']} --image_stride 10")
    print(f"\n  Then run: python process_all_scenes.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()