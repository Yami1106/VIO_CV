"""
generate_all_scenes_stereo.py
------------------------------
Generates all scenes for stereo training.

Usage:
    python generate_all_scenes_stereo.py
"""

import subprocess
import sys
import os

DATASETS_DIR = "/mnt/data/datasets"

# generate_trajectory.py lives alongside this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_SCRIPT = os.path.join(SCRIPT_DIR, "generate_trajectory_stereo.py")

SCENES = [
    # --- TRAIN (7 scenes) ---
    {"name": "scene_001_circle",    "traj": "circle",    "duration": 100, "scale": 3.0},
    {"name": "scene_002_oval",      "traj": "oval",      "duration": 100, "scale": 3.0},
    {"name": "scene_003_diamond",   "traj": "diamond",   "duration": 100, "scale": 3.0},
    {"name": "scene_004_figure8",   "traj": "figure8",   "duration": 100, "scale": 3.0},
    {"name": "scene_005_star",      "traj": "star",      "duration": 100, "scale": 3.0},
    {"name": "scene_006_clover",    "traj": "clover",    "duration": 100, "scale": 3.0},
    {"name": "scene_007_mouse",     "traj": "mouse",     "duration": 100, "scale": 3.0},

    # --- VAL (1 scene) ---
    {"name": "scene_008_halfmoon",  "traj": "halfmoon",  "duration": 100, "scale": 3.0},

    # --- TEST (2 scenes) ---
    {"name": "scene_009_sid",       "traj": "sid",       "duration": 100, "scale": 3.0},
    {"name": "scene_010_picasso",   "traj": "picasso",   "duration": 100, "scale": 3.0},
]


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Generating all scenes → {DATASETS_DIR}")
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
    print("  Done generating trajectories.")
    print()
    print("  Next: render each scene with stereo in Blender:")
    for scene in SCENES:
        print(f"    blender -b YOUR_SCENE.blend -P render_stereo.py -- --scene_name {scene['name']} --stereo --baseline 0.10 --image_stride 10")
    print()
    print("  Then run: python process_all_scenes_stereo.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
