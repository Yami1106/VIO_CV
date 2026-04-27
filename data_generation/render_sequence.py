import bpy
import os
import csv
from mathutils import Vector, Quaternion

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
POSES_CSV = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", "scene_001", "poses.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", "scene_001", "images_test")

CAMERA_NAME = "Camera"
MAX_TEST_FRAMES = 10


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

    # Blender expects quaternion as (w, x, y, z)
    q = Quaternion((pose["qw"], pose["qx"], pose["qy"], pose["qz"]))
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = q


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in bpy.context.scene.render.bl_rna.properties["engine"].enum_items.keys() else "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 320
    scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100


def render_test_sequence():
    ensure_output_dir(OUTPUT_DIR)
    configure_render()

    camera_obj = bpy.data.objects.get(CAMERA_NAME)
    if camera_obj is None:
        raise ValueError(f"Camera object '{CAMERA_NAME}' not found in Blender scene.")

    poses = load_poses(POSES_CSV)
    if len(poses) == 0:
        raise ValueError("No poses found in poses.csv")

    num_frames = min(MAX_TEST_FRAMES, len(poses))

    for i in range(num_frames):
        pose = poses[i]
        set_camera_pose(camera_obj, pose)

        output_path = os.path.join(OUTPUT_DIR, f"image_{i:06d}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        print(f"Rendered {output_path}")


if __name__ == "__main__":
    render_test_sequence()