
import os
import json
import tempfile
import subprocess
from pathlib import Path
from squirrel.library.render.renderer import Renderer


class BlenderRenderer(Renderer):
    """
    Render a Scene using Blender as backend.

    Requires:
        - blender executable in PATH
        - bpy available when running inside Blender
    """

    def __init__(
        self,
        blender_executable="blender",
        template=None,
        output_size=(1200, 1200),
        samples=128,
    ):
        self.blender_executable = blender_executable
        self.template = template
        self.output_size = output_size
        self.samples = samples


    def render(
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
            ]

            subprocess.run(
                cmd,
                check=True,
            )


    def _export_scene(
        self,
        scene,
        export_dir,
    ):

        objects = []

        for i, obj in enumerate(scene.objects):

            mesh_file = export_dir / f"object_{i}.ply"

            obj.mesh.save(mesh_file)

            objects.append(
                {
                    "mesh": str(mesh_file),
                    "color": obj.color,
                    "name": obj.name or f"object_{i}",
                }
            )

        cam = scene.plotter.camera

        return {
            "objects": objects,

            "camera": {
                "position":
                    list(cam.position),

                "focal_point":
                    list(cam.focal_point),

                "up":
                    list(cam.up),
            }
        }


    def _blender_script(self):

        return r'''
import bpy
import sys
import json
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


# -------------------------------------------------
# clean scene
# -------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# -------------------------------------------------
# load scene
# -------------------------------------------------

with open(scene_json) as f:
    data = json.load(f)



# -------------------------------------------------
# materials
# -------------------------------------------------

def make_material(name, color):

    mat = bpy.data.materials.new(name)

    mat.use_nodes = True

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

    return mat



# -------------------------------------------------
# import meshes
# -------------------------------------------------

for obj in data["objects"]:

    bpy.ops.import_mesh.ply(
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

cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new(
    "Camera",
    cam_data,
)

bpy.context.collection.objects.link(cam)

bpy.context.scene.camera = cam


c = data["camera"]

cam.location = c["position"]

target = mathutils.Vector(
    c["focal_point"]
)

direction = (
    target -
    cam.location
)

cam.rotation_euler = (
    direction.to_track_quat(
        "-Z",
        "Y",
    )
    .to_euler()
)



# -------------------------------------------------
# lighting
# -------------------------------------------------

world = bpy.context.scene.world

world.color = (
    0.05,
    0.05,
    0.05,
)


light_data = bpy.data.lights.new(
    "Key",
    type="AREA",
)

light_data.energy = 800
light_data.size = 5


light = bpy.data.objects.new(
    "Key",
    light_data,
)

light.location = (
    5000,
    5000,
    5000,
)

bpy.context.collection.objects.link(light)



# -------------------------------------------------
# renderer
# -------------------------------------------------

scene = bpy.context.scene

scene.render.engine = "BLENDER_EEVEE_NEXT"

scene.render.resolution_x = width
scene.render.resolution_y = height

scene.render.resolution_percentage = 100


scene.render.filepath = output

bpy.ops.render.render(
    write_still=True
)
'''
