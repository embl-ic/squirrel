import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

from squirrel.library.render.renderer import Renderer


class BlenderRenderer(Renderer):
    """
    Render a Scene using Blender.

    The renderer exports all meshes and scene metadata to a temporary
    directory, launches Blender in background mode, imports the scene,
    and renders a still image.
    """

    def __init__(
        self,
        blender_executable="blender",
        template=None,
        output_size=(1200,1200),
        samples=128,
        world_scale=1e-3,
    ):
        self.blender_executable = blender_executable
        self.template = template
        self.output_size = output_size
        self.samples = samples
        self.world_scale = world_scale

    def show(self, scene):
        raise NotImplementedError(
            "Interactive rendering is not supported by BlenderRenderer. "
            "Use screenshot(scene, filename) instead."
        )

    def screenshot(
        self,
        scene,
        filename,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            export_dir = tmp / "meshes"
            export_dir.mkdir()

            scene_file = tmp / "scene.json"

            data = self._export_scene(
                scene,
                export_dir,
            )

            scene_file.write_text(
                json.dumps(data, indent=2)
            )

            script = tmp / "render.py"

            script.write_text(
                self._blender_script()
            )

            cmd = [
                "stdbuf",
                "-oL",
                self.blender_executable,
            ]

            if self.template:
                cmd.append(self.template)

            cmd += [
                "--background",
                "--python",
                str(script),
                "--",
                str(scene_file),
                str(filename),
                str(self.output_size[0]),
                str(self.output_size[1]),
                str(self.samples),
                "render",
            ]

            import subprocess

            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )

            print(process.stdout)
            print(process.stderr)
            print(process.returncode)

    def _export_scene(
        self,
        scene,
        export_dir,
    ):

        from matplotlib.colors import to_rgb

        objects = []
        mins = []
        maxs = []

        for i, obj in enumerate(scene.objects):

            mesh_file = export_dir / f"object_{i}.ply"

            mesh = obj.mesh.copy()
            mesh.points *= self.world_scale

            mins.append(mesh.points.min(axis=0))
            maxs.append(mesh.points.max(axis=0))

            mesh_file = export_dir / f"object_{i}.ply"
            mesh.save(mesh_file)

            objects.append(
                {
                    "mesh": str(mesh_file),
                    "color": list(to_rgb(obj.color)),
                    "name": obj.name or f"object_{i}",
                    "hue_shift": obj.hue_shift,
                    "lightness_shift": obj.lightness_shift,
                    "normal_mode": obj.normal_mode,
                }
            )

        mins = np.vstack(mins).min(axis=0)
        maxs = np.vstack(maxs).max(axis=0)

        center = (mins + maxs) / 2
        extent = maxs - mins

        if scene.camera_position is None:
            raise RuntimeError(
                "BlenderRenderer requires scene.camera_position."
            )

        camera = scene.camera_position

        return {
            "objects": objects,

            "camera": {
                "position": (
                    np.array(camera[0]) * self.world_scale
                ).tolist(),

                "focal_point": (
                    np.array(camera[1]) * self.world_scale
                ).tolist(),

                "up": list(camera[2]),

                "view_angle": 30.0,
            },

            "background": scene.background,

            "bounds": {
                "center": center.tolist(),
                "extent": extent.tolist(),
            },
        }

    def _blender_script(self, save_only=False):

        return r'''
import bpy
import sys
import json
import math
import mathutils


# -------------------------------------------------
# arguments
# -------------------------------------------------

argv = sys.argv

argv = argv[
    argv.index("--") + 1:
]

scene_json = argv[0]
output = argv[1]

width = int(argv[2])
height = int(argv[3])
samples = int(argv[4])
mode = argv[5]
save_only = (mode == "save")

# -------------------------------------------------
# clean scene
# -------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# -------------------------------------------------
# load scene data
# -------------------------------------------------

with open(scene_json) as f:
    data = json.load(f)

# -------------------------------------------------
# materials
# -------------------------------------------------

def make_material(name, color):

    mat = bpy.data.materials.new(name)

    bsdf = (
        mat.node_tree
        .nodes
        .get("Principled BSDF")
    )

    bsdf.inputs["Base Color"].default_value = (
        color[0],
        color[1],
        color[2],
        1.0,
    )

    bsdf.inputs["Roughness"].default_value = 0.7

    bsdf.inputs["Emission Color"].default_value = (
        color[0],
        color[1],
        color[2],
        1.0,
    )

    bsdf.inputs["Emission Strength"].default_value = 0.0

    return mat


# -------------------------------------------------
# import meshes
# -------------------------------------------------

for obj in data["objects"]:

    bpy.ops.wm.ply_import(
        filepath=obj["mesh"]
    )

    mesh_obj = bpy.context.object

    mesh_obj.name = obj["name"]

    mat = make_material(
        obj["name"],
        obj["color"],
    )

    mesh_obj.data.materials.append(mat)

# -------------------------------------------------
# camera
# -------------------------------------------------

# -------------------------------------------------
# camera
# -------------------------------------------------

c = data["camera"]

cam_data = bpy.data.cameras.new("Camera")
cam_data.angle = math.radians(c["view_angle"])

cam = bpy.data.objects.new(
    "Camera",
    cam_data,
)

bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

cam.location = c["position"]

target = mathutils.Vector(c["focal_point"])

direction = target - cam.location

cam.rotation_euler = (
    direction.to_track_quat("-Z", "Y").to_euler()
)

cam_data.clip_start = 0.1
cam_data.clip_end = 100000

# -------------------------------------------------
# lighting
# -------------------------------------------------

world = bpy.context.scene.world

if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

# world.use_nodes = True

bg = world.node_tree.nodes.get("Background")

bg.inputs["Color"].default_value = (
    1.0,
    1.0,
    1.0,
    1.0,
)

bg.inputs["Strength"].default_value = 0.08

# Add area light

bounds = data["bounds"]

center = mathutils.Vector(bounds["center"])
extent = mathutils.Vector(bounds["extent"])

diameter = max(extent)

light_data = bpy.data.lights.new(
    "Key",
    type="AREA",
)

light_data.energy = 150
light_data.shape = "SQUARE"
light_data.size = 2.0 * diameter

light = bpy.data.objects.new(
    "Key",
    light_data,
)

light.location = center + mathutils.Vector((
    diameter,
    diameter,
    1.5 * diameter,
))

bpy.context.collection.objects.link(light)

light.rotation_euler = (
    (center - light.location)
    .to_track_quat("-Z", "Y")
    .to_euler()
)

# -------------------------------------------------
# renderer settings
# -------------------------------------------------

scene = bpy.context.scene

scene.render.engine = "CYCLES"

scene.cycles.samples = samples
scene.cycles.use_denoising = True

# scene.render.engine = "BLENDER_EEVEE"

scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100

# Color handling
# scene.view_settings.look = "AgX - Medium High Contrast"
# scene.view_settings.exposure = 3.0
scene.view_settings.look = "AgX - High Contrast"
scene.view_settings.exposure = 2.2
scene.view_settings.gamma = 0.9

# Sampling
if scene.render.engine == "BLENDER_EEVEE":
    scene.eevee.taa_render_samples = samples


# Transparent background disabled
scene.render.film_transparent = True


scene.render.filepath = output


# -------------------------------------------------
# render
# -------------------------------------------------

if save_only:

    bpy.ops.wm.save_as_mainfile(
        filepath=output
    )

else:

    bpy.ops.render.render(
        write_still=True
)
'''

    def write_blend(
        self,
        scene,
        filename,
    ):
        """
        Export the scene as a Blender .blend file.

        The resulting file can be opened interactively in Blender
        for lighting, material, camera, and render adjustments.
        """

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            export_dir = tmp / "meshes"
            export_dir.mkdir()

            scene_file = tmp / "scene.json"

            data = self._export_scene(
                scene,
                export_dir,
            )

            scene_file.write_text(
                json.dumps(data, indent=2)
            )

            script = tmp / "render.py"

            script.write_text(
                self._blender_script(save_only=True)
            )

            cmd = [
                self.blender_executable,
            ]

            if self.template:
                cmd.append(self.template)

            cmd += [
                "--background",
                "--python",
                str(script),
                "--",
                str(scene_file),
                str(filename),
                str(self.output_size[0]),
                str(self.output_size[1]),
                str(self.samples),
                "save",
            ]

            import subprocess

            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )

            print(process.stdout)
            print(process.stderr)
            print(process.returncode)
            