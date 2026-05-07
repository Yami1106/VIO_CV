"""
render_stereo.py — Blender stereo rendering

Usage (from Blender):
    blender -b YOUR_SCENE.blend -P render_stereo.py -- --scene_name scene_001_circle --stereo --baseline 0.10 --image_stride 10
"""

import bpy
import os
import csv
import argparse
from mathutils import Vector, Quaternion

DATASETS_DIR = "/mnt/data/datasets"
CAMERA_NAME = "Camera"
RIGHT_CAMERA_NAME = "Camera_Right"


def load_poses(csv_path):
    poses = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "timestamp": float(row["timestamp"]),
                "px": float(row["px"]),
                "py": float(row["py"]),
                "pz": float(row["pz"]),
                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "qw": float(row["qw"]),
            })
    return poses


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def set_camera_pose(camera_obj, pose):
    camera_obj.location = Vector((pose["px"], pose["py"], pose["pz"]))
    q = Quaternion((pose["qw"], pose["qx"], pose["qy"], pose["qz"]))
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = q


def get_or_create_right_camera(left_camera_obj):
    right_camera_obj = bpy.data.objects.get(RIGHT_CAMERA_NAME)

    if right_camera_obj is None:
        right_camera_data = left_camera_obj.data.copy()
        right_camera_data.name = RIGHT_CAMERA_NAME + "_Data"
        right_camera_obj = bpy.data.objects.new(RIGHT_CAMERA_NAME, right_camera_data)
        bpy.context.collection.objects.link(right_camera_obj)

    right_camera_obj.data.lens = left_camera_obj.data.lens
    right_camera_obj.data.sensor_width = left_camera_obj.data.sensor_width
    right_camera_obj.data.sensor_height = left_camera_obj.data.sensor_height
    right_camera_obj.data.angle = left_camera_obj.data.angle
    right_camera_obj.data.clip_start = left_camera_obj.data.clip_start
    right_camera_obj.data.clip_end = left_camera_obj.data.clip_end

    return right_camera_obj


def set_stereo_camera_poses(left_camera_obj, right_camera_obj, pose, baseline):
    q = Quaternion((pose["qw"], pose["qx"], pose["qy"], pose["qz"]))

    left_camera_obj.location = Vector((pose["px"], pose["py"], pose["pz"]))
    left_camera_obj.rotation_mode = "QUATERNION"
    left_camera_obj.rotation_quaternion = q

    right_camera_obj.rotation_mode = "QUATERNION"
    right_camera_obj.rotation_quaternion = q

    right_offset_world = q @ Vector((baseline, 0.0, 0.0))
    right_camera_obj.location = left_camera_obj.location + right_offset_world


def configure_render(resolution_x=320, resolution_y=320):
    scene = bpy.context.scene

    # --- Force GPU rendering via Cycles ---
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    # --- Enable GPU devices ---
    # Must be done AFTER setting engine to CYCLES
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences

        # Try each compute backend
        for compute_type in ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = compute_type
                break
            except TypeError:
                continue

        # Force device refresh
        prefs.get_devices()

        # Enable ALL devices (GPU + CPU fallback)
        gpu_found = False
        for device in prefs.devices:
            device.use = True
            if device.type != "CPU":
                gpu_found = True
            print(f"  Device: {device.name} ({device.type}) — enabled")

        if not gpu_found:
            print("  WARNING: No GPU device found, falling back to CPU!")
            scene.cycles.device = "CPU"
        else:
            print(f"  GPU rendering enabled: {prefs.compute_device_type}")

    except Exception as e:
        print(f"  WARNING: Could not configure GPU: {e}")
        print(f"  Falling back to EEVEE (CPU)")
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "BLENDER_EEVEE"

    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100


def render_one_image(camera_obj, output_path):
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_sequence(
    poses_csv,
    output_dir,
    max_frames=None,
    resolution_x=320,
    resolution_y=320,
    image_stride=10,
    stereo=False,
    baseline=0.10
):
    ensure_output_dir(output_dir)
    configure_render(resolution_x=resolution_x, resolution_y=resolution_y)

    camera_obj = bpy.data.objects.get(CAMERA_NAME)
    if camera_obj is None:
        raise ValueError(f"Camera object '{CAMERA_NAME}' not found in Blender scene.")

    right_camera_obj = None
    if stereo:
        right_camera_obj = get_or_create_right_camera(camera_obj)
        left_output_dir = os.path.join(output_dir, "left")
        right_output_dir = os.path.join(output_dir, "right")
        ensure_output_dir(left_output_dir)
        ensure_output_dir(right_output_dir)
    else:
        ensure_output_dir(output_dir)

    poses = load_poses(poses_csv)
    if len(poses) == 0:
        raise ValueError("No poses found in poses.csv")

    num_frames = len(poses) if max_frames is None else min(max_frames, len(poses))

    saved_idx = 0
    for i in range(0, num_frames, image_stride):
        pose = poses[i]

        if stereo:
            set_stereo_camera_poses(
                left_camera_obj=camera_obj,
                right_camera_obj=right_camera_obj,
                pose=pose,
                baseline=baseline
            )

            left_output_path = os.path.join(left_output_dir, f"image_{saved_idx:06d}.png")
            right_output_path = os.path.join(right_output_dir, f"image_{saved_idx:06d}.png")

            render_one_image(camera_obj, left_output_path)
            render_one_image(right_camera_obj, right_output_path)

            print(f"Rendered stereo pair {saved_idx:06d} from pose index {i}")

        else:
            set_camera_pose(camera_obj, pose)

            output_path = os.path.join(output_dir, f"image_{saved_idx:06d}.png")
            render_one_image(camera_obj, output_path)

            print(f"Rendered {output_path} from pose index {i}")

        saved_idx += 1


def main():
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resolution_x", type=int, default=320)
    parser.add_argument("--resolution_y", type=int, default=320)
    parser.add_argument("--image_stride", type=int, default=10)
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--baseline", type=float, default=0.10)
    args = parser.parse_args(argv)

    scene_dir = os.path.join(DATASETS_DIR, args.scene_name)
    poses_csv = os.path.join(scene_dir, "poses.csv")
    output_dir = os.path.join(scene_dir, "images")

    render_sequence(
        poses_csv=poses_csv,
        output_dir=output_dir,
        max_frames=args.max_frames,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        image_stride=args.image_stride,
        stereo=args.stereo,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()