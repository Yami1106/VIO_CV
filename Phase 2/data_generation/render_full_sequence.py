import bpy
import os
import csv
import argparse
from mathutils import Vector, Quaternion

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
CAMERA_NAME = "Camera"


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


def configure_render(resolution_x=320, resolution_y=320):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100


def render_sequence(poses_csv, output_dir, max_frames=None, resolution_x=320, resolution_y=320, image_stride=10):
    ensure_output_dir(output_dir)
    configure_render(resolution_x=resolution_x, resolution_y=resolution_y)

    camera_obj = bpy.data.objects.get(CAMERA_NAME)
    if camera_obj is None:
        raise ValueError(f"Camera object '{CAMERA_NAME}' not found in Blender scene.")

    poses = load_poses(poses_csv)
    if len(poses) == 0:
        raise ValueError("No poses found in poses.csv")

    num_frames = len(poses) if max_frames is None else min(max_frames, len(poses))

    saved_idx = 0
    for i in range(0, num_frames, image_stride):
        pose = poses[i]
        set_camera_pose(camera_obj, pose)

        output_path = os.path.join(output_dir, f"image_{saved_idx:06d}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

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
    args = parser.parse_args(argv)

    scene_dir = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", args.scene_name)
    poses_csv = os.path.join(scene_dir, "poses.csv")
    output_dir = os.path.join(scene_dir, "images")

    render_sequence(
        poses_csv=poses_csv,
        output_dir=output_dir,
        max_frames=args.max_frames,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        image_stride=args.image_stride,
    )


if __name__ == "__main__":
    main()