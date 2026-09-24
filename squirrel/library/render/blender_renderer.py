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

        filename = Path(filename)
        blend_file = filename.with_suffix(".blend")

        self.write_blend(
            scene,
            blend_file,
        )

        self.render_blend(
            blend_file,
            filename,
        )

    def render_blend(
        self,
        blend_file,
        output_file,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            script = tmp / "render.py"

            script.write_text(
                self._render_script()
            )

            cmd = [
                self.blender_executable,
                str(blend_file),
                "--background",
                "--python",
                str(script),
                "--",
                str(output_file),
                str(self.output_size[0]),
                str(self.output_size[1]),
                str(self.samples),
            ]

            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )

            print("========== STDOUT ==========")
            print(process.stdout)

            print("========== STDERR ==========")
            print(process.stderr)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Blender failed with exit code {process.returncode}"
                )

    def _export_scene(
        self,
        scene,
        export_dir,
    ):

        from matplotlib.colors import to_rgb
        from PIL import Image

        objects = []
        slices = []
        mins = []
        maxs = []

        # -------------------------------------------------
        # Objects
        # -------------------------------------------------

        for i, obj in enumerate(scene.objects):

            mesh_file = export_dir / f"object_{i}.ply"

            mesh = obj.mesh.copy()
            mesh.points *= self.world_scale

            mins.append(mesh.points.min(axis=0))
            maxs.append(mesh.points.max(axis=0))

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

        # -------------------------------------------------
        # EM slices
        # -------------------------------------------------

        for i, em_slice in enumerate(scene.slices):

            volume = em_slice.volume
            data = volume.data

            spacing = (
                np.asarray(volume.voxel_size, dtype=float)
                * self.world_scale
            )
            volume_origin = (
                np.asarray(volume.origin, dtype=float)
                * self.world_scale
            )
            sz, sy, sx = spacing
            oz, oy, ox = volume_origin

            nz, ny, nx = data.shape

            axis = em_slice.axis
            index = em_slice.index

            if axis == "z":
                image = data[index, :, :]
                origin = [ox, oy, oz + index * sz]
                size = [(nx - 1) * sx, (ny - 1) * sy]

            elif axis == "y":
                image = data[:, index, :]
                origin = [ox, oy + index * sy, oz]
                size = [(nx - 1) * sx, (nz - 1) * sz]

            elif axis == "x":
                image = data[:, :, index]
                origin = [ox + index * sx, oy, oz]
                size = [(ny - 1) * sy, (nz - 1) * sz]
            else:
                raise ValueError(
                    f"Invalid slice axis: {axis}"
                )

            # ---------------------------------------------
            # Convert EM data to uint8
            # ---------------------------------------------

            if em_slice.clim is None:
                vmin = float(np.min(image))
                vmax = float(np.max(image))
            else:
                vmin, vmax = em_slice.clim

            denom = vmax - vmin

            if denom == 0:
                image_normalized = np.zeros_like(
                    image,
                    dtype=np.float32,
                )
            else:
                image_normalized = np.clip(
                    (image.astype(np.float32) - vmin) / denom,
                    0,
                    1,
                )

            image_uint8 = (
                image_normalized * 255
            ).astype(np.uint8)

            image_file = (
                export_dir / f"slice_{i}.png"
            )

            Image.fromarray(image_uint8).save(
                image_file
            )

            slices.append(
                {
                    "image": str(image_file),
                    "axis": axis,
                    "index": int(index),
                    "origin": origin,
                    "size": size,
                    "opacity": em_slice.opacity,
                }
            )

        # -------------------------------------------------
        # Bounds
        # -------------------------------------------------

        # Use the underlying data bounds so empty segmentations and EM-only
        # scenes still have deterministic framing.
        bounds_min, bounds_max = scene.get_bounds()
        mins = np.asarray(bounds_min, dtype=float) * self.world_scale
        maxs = np.asarray(bounds_max, dtype=float) * self.world_scale

        center = (mins + maxs) / 2
        extent = maxs - mins

        if scene.camera_position is None:
            raise RuntimeError(
                "BlenderRenderer requires scene.camera_position."
            )

        camera = scene.camera_position

        camera_projection = scene.camera_projection
        ortho_scale = None
        camera_position = np.asarray(camera[0], dtype=float) * self.world_scale
        camera_focal = np.asarray(camera[1], dtype=float) * self.world_scale
        camera_up = np.asarray(camera[2], dtype=float)
        view_angle = 30.0

        if (
            scene.camera_preset is not None
            and camera_projection == "orthographic"
        ):
            axis = scene.camera_preset[0]

            if axis == "z":
                width = extent[0]   # world x
                height = extent[1]  # world y
            elif axis == "y":
                width = extent[0]   # world x
                height = extent[2]  # world z
            elif axis == "x":
                width = extent[1]   # world y
                height = extent[2]  # world z
            else:
                raise ValueError(
                    f"Invalid camera preset: {scene.camera_preset}"
                )

            aspect = self.output_size[0] / self.output_size[1]

            # Blender's ortho_scale is the horizontal camera-frame width.
            # Ensure both projected width and height fit the output aspect.
            ortho_scale = max(
                width,
                height * aspect,
            ) * (1.0 + scene.camera_fit_padding)

        elif (
            scene.camera_preset is not None
            and camera_projection == "perspective"
        ):
            camera_side = camera_position - camera_focal
            camera_side /= np.linalg.norm(camera_side)

            forward = -camera_side
            camera_up /= np.linalg.norm(camera_up)
            right = np.cross(forward, camera_up)
            right /= np.linalg.norm(right)
            camera_up = np.cross(right, forward)
            camera_up /= np.linalg.norm(camera_up)

            corners = np.array([
                [x, y, z]
                for x in (mins[0], maxs[0])
                for y in (mins[1], maxs[1])
                for z in (mins[2], maxs[2])
            ])
            offsets = corners - camera_focal

            aspect = self.output_size[0] / self.output_size[1]
            tan_v = np.tan(np.deg2rad(view_angle) * 0.5)
            tan_h = tan_v * aspect
            padding = 1.0 + scene.camera_fit_padding

            along = offsets @ camera_side
            horizontal = np.abs(offsets @ right)
            vertical = np.abs(offsets @ camera_up)

            required_distance = np.max(
                along
                + np.maximum(
                    padding * horizontal / tan_h,
                    padding * vertical / tan_v,
                )
            )

            camera_position = (
                camera_focal
                + camera_side * required_distance
            )

        elif camera_projection not in {"perspective", "orthographic"}:
            raise ValueError(
                f"Invalid camera projection: {camera_projection}"
            )

        return {
            "objects": objects,
            "slices": slices,

            "camera": {
                "position": camera_position.tolist(),
                "focal_point": camera_focal.tolist(),
                "up": camera_up.tolist(),

                "view_angle": view_angle,
                "projection": camera_projection,
                "ortho_scale": ortho_scale,
                "flip_horizontal": scene.camera_flip_horizontal,
            },

            "background": scene.background,

            "bounds": {
                "center": center.tolist(),
                "extent": extent.tolist(),
            },
        }

    def _build_scene_script(self):

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
# width = int(argv[2])
# height = int(argv[3])
# samples = int(argv[4])

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
# EM slice materials
# -------------------------------------------------

def make_slice_material(name, image_file, opacity=1.0):

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new(
        "ShaderNodeOutputMaterial"
    )

    bsdf = nodes.new(
        "ShaderNodeBsdfPrincipled"
    )

    tex = nodes.new(
        "ShaderNodeTexImage"
    )

    image = bpy.data.images.load(image_file)
    image.pack()

    tex.image = image
    tex.interpolation = "Linear"

    # Use the EM image as the diffuse/base color so the slice participates
    # in normal scene lighting and can receive/cast Cycles shadows.
    links.new(
        tex.outputs["Color"],
        bsdf.inputs["Base Color"],
    )

    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0

    # Add a small self-illuminated contribution from the same image. This
    # keeps EM contrast readable without making the plane fully emissive,
    # which would suppress the visual effect of lighting and shadows.
    links.new(
        tex.outputs["Color"],
        bsdf.inputs["Emission Color"],
    )
    bsdf.inputs["Emission Strength"].default_value = 0.05

    bsdf.inputs["Alpha"].default_value = opacity

    links.new(
        bsdf.outputs["BSDF"],
        output.inputs["Surface"],
    )

    return mat

# -------------------------------------------------
# EM slices
# -------------------------------------------------

for i, s in enumerate(data.get("slices", [])):

    axis = s["axis"]
    origin = s["origin"]
    size = s["size"]

    # -------------------------------------------------
    # Explicit geometry
    #
    # Library/world coordinates are conventional Blender coordinates:
    #   world X = numpy x, world Y = numpy y, world Z = numpy z.
    # size stores the two in-plane dimensions in world-axis order.

    if axis == "z":
        x0, y0, z = origin
        sx, sy = size
        vertices = [
            (x0,      y0,      z),
            (x0 + sx, y0,      z),
            (x0 + sx, y0 + sy, z),
            (x0,      y0 + sy, z),
        ]

    elif axis == "y":
        x0, y, z0 = origin
        sx, sz = size
        vertices = [
            (x0,      y, z0),
            (x0 + sx, y, z0),
            (x0 + sx, y, z0 + sz),
            (x0,      y, z0 + sz),
        ]

    elif axis == "x":
        x, y0, z0 = origin
        sy, sz = size
        vertices = [
            (x, y0,      z0),
            (x, y0 + sy, z0),
            (x, y0 + sy, z0 + sz),
            (x, y0,      z0 + sz),
        ]

    else:
        raise ValueError(
            f"Invalid slice axis: {axis}"
        )

    faces = [
        (0, 1, 2, 3),
    ]

    mesh = bpy.data.meshes.new(
        f"EM_slice_mesh_{i}"
    )

    mesh.from_pydata(
        vertices,
        [],
        faces,
    )

    mesh.update()

    plane = bpy.data.objects.new(
        f"EM_slice_{axis}_{s['index']}",
        mesh,
    )

    bpy.context.collection.objects.link(
        plane
    )

    # -------------------------------------------------
    # UV coordinates
    # -------------------------------------------------

    uv_layer = mesh.uv_layers.new(
        name="UVMap"
    )

    # PIL/NumPy row 0 is at the top of the image,
    # while Blender V=0 is the bottom. Therefore
    # V is deliberately flipped here.
    uv_coords = [
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),
    ]

    for loop, uv in zip(
        mesh.loops,
        uv_coords,
    ):
        uv_layer.data[loop.index].uv = uv

    # -------------------------------------------------
    # Material
    # -------------------------------------------------

    mat = make_slice_material(
        f"EM_slice_material_{i}",
        s["image"],
        s["opacity"],
    )

    plane.data.materials.append(mat)

# -------------------------------------------------
# camera
# -------------------------------------------------

c = data["camera"]

cam_data = bpy.data.cameras.new("Camera")

# Projection is exported explicitly by Scene. Perspective is the default.
projection = c.get("projection", "perspective").lower()

if projection == "perspective":
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = 32.0
    cam_data.lens = (
        0.5 * cam_data.sensor_height
        / math.tan(0.5 * math.radians(c["view_angle"]))
    )
elif projection == "orthographic":
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = c["ortho_scale"]
else:
    raise ValueError(f"Invalid camera projection: {projection}")

cam = bpy.data.objects.new(
    "Camera",
    cam_data,
)

bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

cam.location = c["position"]

target = mathutils.Vector(c["focal_point"])

forward = (target - cam.location).normalized()
up = mathutils.Vector(c["up"]).normalized()

# Blender cameras look along local -Z with local +Y as camera-up.
# Build the camera basis explicitly so Scene.view_up is preserved.
right = forward.cross(up).normalized()
up = right.cross(forward).normalized()

rotation = mathutils.Matrix((
    right,
    up,
    -forward,
)).transposed()

cam.rotation_euler = rotation.to_euler()

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

# Keep the key light on roughly the same side as the camera, but offset it
# in camera-space so illumination is not head-on and still produces shadows.
light.location = (
    center
    - forward * (1.5 * diameter)
    + right * (0.7 * diameter)
    + up * (0.8 * diameter)
)

bpy.context.collection.objects.link(light)

light.rotation_euler = (
    (center - light.location)
    .to_track_quat("-Z", "Y")
    .to_euler()
)

# -------------------------------------------------
# render
# -------------------------------------------------

scene = bpy.context.scene
scene.render.engine = "CYCLES"

# Store the intended color-management settings in the .blend itself so the
# interactive Blender project matches the final scripted render.
scene.view_settings.look = "AgX - High Contrast"
scene.view_settings.exposure = 2.2
scene.view_settings.gamma = 0.9

# scene.render.resolution_x = width
# scene.render.resolution_y = height
# scene.cycles.samples = samples

bpy.ops.wm.save_as_mainfile(
    filepath=output
)
'''

    def _render_script(self):

        return r'''
import bpy
import sys

argv = sys.argv[sys.argv.index("--")+1:]

output = argv[0]
width = int(argv[1])
height = int(argv[2])
samples = int(argv[3])

scene = bpy.context.scene

# -------------------------------------------------
# Cycles GPU configuration
# -------------------------------------------------

scene.render.engine = "CYCLES"

prefs = bpy.context.preferences

cycles_prefs = prefs.addons["cycles"].preferences

# Try CUDA first, then OPTIX, then HIP
available_devices = []

for backend in ["OPTIX", "CUDA", "HIP", "METAL"]:
    try:
        cycles_prefs.compute_device_type = backend
        cycles_prefs.get_devices()

        devices = cycles_prefs.devices

        if any(d.type != "CPU" for d in devices):
            print(f"Using Cycles backend: {backend}")

            for device in devices:
                device.use = device.type != "CPU"
                print(
                    f"Device: {device.name} "
                    f"type={device.type} "
                    f"enabled={device.use}"
                )

            scene.cycles.device = "GPU"
            available_devices = devices
            break

    except Exception as e:
        print(
            f"Backend {backend} unavailable: {e}"
        )


if not available_devices:
    print("WARNING: No GPU found, falling back to CPU")
    scene.cycles.device = "CPU"


# -------------------------------------------------
# Render settings
# -------------------------------------------------

scene.cycles.samples = samples
scene.cycles.use_denoising = True

scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100

scene.view_settings.look = "AgX - High Contrast"
scene.view_settings.exposure = 2.2
scene.view_settings.gamma = 0.9

scene.render.film_transparent = True

scene.render.filepath = output

print(
    "Final Cycles device:",
    scene.cycles.device
)

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
                self._build_scene_script()
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
            ]

            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
            )

            print("========== STDOUT ==========")
            print(process.stdout)

            print("========== STDERR ==========")
            print(process.stderr)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Blender failed with exit code {process.returncode}"
                )
            