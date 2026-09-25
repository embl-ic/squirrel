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
        view_transform="Standard",
        look="None",
        exposure=1.5,
        gamma=0.8,
        light_placement="top-right",
        light_angle=35.0,
        light_power=300.0,
        light_temperature=None,
        world_strength=0.5,
        em_slice_emission_strength=0.05,
        upright_scene=False,
    ):
        self.blender_executable = blender_executable
        self.template = template
        self.output_size = output_size
        self.samples = samples
        self.world_scale = world_scale

        self.view_transform = view_transform
        self.look = look
        self.exposure = exposure
        self.gamma = gamma

        valid_placements = {
            "top-left", "top", "top-right",
            "left", "center", "right",
            "bottom-left", "bottom", "bottom-right",
        }
        if light_placement not in valid_placements:
            raise ValueError(
                f"Invalid light_placement: {light_placement}. "
                f"Expected one of {sorted(valid_placements)}."
            )
        if not 0 <= light_angle < 90:
            raise ValueError("light_angle must be in the range [0, 90).")

        self.light_placement = light_placement
        self.light_angle = float(light_angle)
        self.light_power = float(light_power)
        self.light_temperature = light_temperature
        self.world_strength = float(world_strength)
        self.em_slice_emission_strength = float(em_slice_emission_strength)
        self.upright_scene = bool(upright_scene)

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

        filename = Path(filename).expanduser().resolve()
        filename.parent.mkdir(parents=True, exist_ok=True)
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

        blend_file = Path(blend_file).expanduser().resolve()
        output_file = Path(output_file).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

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

            if obj.mesh is None or obj.mesh.n_faces_strict == 0:
                continue

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
                    "interpolation": em_slice.interpolation,
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
                "preset": scene.camera_preset,
            },

            "background": scene.background,

            "render_settings": {
                "view_transform": self.view_transform,
                "look": self.look,
                "exposure": self.exposure,
                "gamma": self.gamma,
                "world_strength": self.world_strength,
                "em_slice_emission_strength": self.em_slice_emission_strength,
                "light_placement": self.light_placement,
                "light_angle": self.light_angle,
                "light_power": self.light_power,
                "light_temperature": self.light_temperature,
                "upright_scene": self.upright_scene,
                "output_size": list(self.output_size),
                "samples": self.samples,
            },

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

def srgb_to_linear(value):
    # Palette colors are supplied as sRGB values (e.g. from a hex color).
    # Blender shader sockets store scene-linear RGB, while Blender's hex
    # color field displays the corresponding sRGB value.
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def make_material(name, color):

    mat = bpy.data.materials.new(name)

    bsdf = (
        mat.node_tree
        .nodes
        .get("Principled BSDF")
    )

    color_linear = tuple(
        srgb_to_linear(channel)
        for channel in color[:3]
    )

    bsdf.inputs["Base Color"].default_value = (
        color_linear[0],
        color_linear[1],
        color_linear[2],
        1.0,
    )

    bsdf.inputs["Roughness"].default_value = 0.7

    bsdf.inputs["Emission Color"].default_value = (
        color_linear[0],
        color_linear[1],
        color_linear[2],
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

def make_slice_material(name, image_file, opacity=1.0, interpolation="linear"):

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

    if interpolation == "nearest":
        tex.interpolation = "Closest"
    elif interpolation == "linear":
        tex.interpolation = "Linear"
    else:
        raise ValueError(f"Invalid EM slice interpolation: {interpolation}")

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
    bsdf.inputs["Emission Strength"].default_value = data["render_settings"]["em_slice_emission_strength"]

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
        s.get("interpolation", "linear"),
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
# Track camera-up explicitly.  Do not recover it later from matrix_world:
# Blender's object transform and our scene reflection are separate concerns.
final_camera_up = up.copy()

rotation = mathutils.Matrix((
    right,
    up,
    -forward,
)).transposed()

cam.rotation_euler = rotation.to_euler()

# Keep the Blender camera transform conventional.  Horizontal reflection,
# when requested, is applied to the exported scene geometry below instead of
# using a negative camera scale.  A reflected camera transform confuses
# Blender's camera/viewport navigation.
cam.scale = (1.0, 1.0, 1.0)

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

settings = data["render_settings"]

bg.inputs["Strength"].default_value = settings["world_strength"]

# Light creation is intentionally deferred until after any upright-scene
# transform/reflection.  Placement semantics (top-left, right, etc.) are
# screen-relative, so they must be evaluated in the final camera basis.

bounds = data["bounds"]
center = mathutils.Vector(bounds["center"])
extent = mathutils.Vector(bounds["extent"])
diameter = max(extent)
final_center = center.copy()

# -------------------------------------------------
# Optional Blender workspace orientation
# -------------------------------------------------

# Keep the Blender workspace axis-aligned.  The upright transform is derived
# only from the canonical camera preset, never from the final (possibly
# orbited) camera orientation.  This means Scene.rotate_camera() remains a
# camera movement and cannot tilt the exported specimen.
if settings.get("upright_scene", False):
    preset = c.get("preset")
    if preset is None:
        raise RuntimeError(
            "upright_scene=True requires a camera preset. "
            "Call scene.set_camera_preset(...) before rendering."
        )

    # Canonical preset up vectors, in library/world coordinates.  Use exact
    # axis rotations so the specimen remains aligned to Blender's axes.
    if preset in {"x_front", "x_back", "z_front", "z_back"}:
        # (0, -1, 0) -> Blender +Z
        upright_rotation = mathutils.Matrix.Rotation(
            math.radians(-90.0), 3, "X"
        )
    elif preset in {"y_front", "y_back"}:
        # (0, 0, -1) -> Blender +Z
        upright_rotation = mathutils.Matrix.Rotation(
            math.radians(180.0), 3, "X"
        )
    else:
        raise ValueError(f"Unsupported camera preset for upright scene: {preset}")

    # Rotate the complete source bounds and then translate vertically so the
    # lowest point of the specimen sits exactly on Blender's XY floor (Z=0).
    b_center = mathutils.Vector(bounds["center"])
    b_extent = mathutils.Vector(bounds["extent"])
    b_min = b_center - 0.5 * b_extent
    b_max = b_center + 0.5 * b_extent

    corners = [
        upright_rotation @ mathutils.Vector((x, y, z))
        for x in (b_min.x, b_max.x)
        for y in (b_min.y, b_max.y)
        for z in (b_min.z, b_max.z)
    ]
    floor_offset = mathutils.Vector((0.0, 0.0, -min(v.z for v in corners)))

    upright_matrix = upright_rotation.to_4x4()
    translation_matrix = mathutils.Matrix.Translation(floor_offset)
    world_transform = translation_matrix @ upright_matrix
    final_center = world_transform @ center

    # Geometry receives one common rigid world transform.  The key light is
    # created later, directly in the final camera/world coordinate system.
    for obj in list(bpy.context.scene.objects):
        if obj is not cam:
            obj.matrix_world = world_transform @ obj.matrix_world

    # Transform the camera position/focal point explicitly.  Its up vector is
    # rotated but, correctly, not translated.  Rebuilding the basis avoids
    # decomposing a reflected camera matrix.
    cam_position = (
        upright_rotation @ mathutils.Vector(c["position"])
        + floor_offset
    )
    cam_target = (
        upright_rotation @ mathutils.Vector(c["focal_point"])
        + floor_offset
    )
    target = cam_target
    cam_up = (
        upright_rotation @ mathutils.Vector(c["up"])
    ).normalized()

    cam.location = cam_position

    cam_forward = (cam_target - cam.location).normalized()
    cam_right = cam_forward.cross(cam_up).normalized()
    cam_up = cam_right.cross(cam_forward).normalized()
    final_camera_up = cam_up.copy()

    cam_rotation = mathutils.Matrix((
        cam_right,
        cam_up,
        -cam_forward,
    )).transposed()

    cam.rotation_euler = cam_rotation.to_euler()
    cam.scale = (1.0, 1.0, 1.0)

    # Canonical preset direction after the upright transform.  This is used
    # only for the saved working viewport; it is independent of camera orbit.
    preset_view_dirs = {
        "x_front": (-1.0, 0.0, 0.0),
        "x_back":  ( 1.0, 0.0, 0.0),
        "y_front": (0.0, -1.0, 0.0),
        "y_back":  (0.0,  1.0, 0.0),
        "z_front": (0.0, 0.0, -1.0),
        "z_back":  (0.0, 0.0,  1.0),
    }
    canonical_forward = (
        upright_rotation @ mathutils.Vector(preset_view_dirs[preset])
    ).normalized()

    # Save a useful non-camera viewport as well.  It is aligned to the
    # canonical preset side (front/back), while NUM0 still enters the exact
    # render camera including any Scene.rotate_camera() orbit.
    # RegionView3D uses a view quaternion, not a camera object's transform.
    # Build it from Blender's standard viewing convention directly: local -Z
    # looks toward the specimen and local +Y is visual up.  Keeping +Z as the
    # canonical workspace up prevents the saved viewport from opening on its
    # side even though the render camera may be oblique.
    # Use Blender's *native standard-view quaternions* instead of converting
    # the render-camera basis into RegionView3D coordinates.  RegionView3D's
    # quaternion convention is different from a camera object's transform;
    # using a camera-style track quaternion here is what caused the saved
    # viewport to open with a 90-degree roll.
    #
    # canonical_forward points from the viewer toward the specimen.  After
    # the upright transform it is axis-aligned, so select the corresponding
    # Blender standard view (the same orientations used by the numpad views).
    sqrt_half = 2.0 ** -0.5
    standard_views = {
        # view type: (forward direction, RegionView3D.view_rotation)
        "FRONT":  (mathutils.Vector(( 0.0,  1.0,  0.0)), mathutils.Quaternion((sqrt_half, sqrt_half, 0.0, 0.0))),
        "BACK":   (mathutils.Vector(( 0.0, -1.0,  0.0)), mathutils.Quaternion((0.0, 0.0, sqrt_half, sqrt_half))),
        "RIGHT":  (mathutils.Vector((-1.0,  0.0,  0.0)), mathutils.Quaternion((0.5, 0.5, 0.5, 0.5))),
        "LEFT":   (mathutils.Vector(( 1.0,  0.0,  0.0)), mathutils.Quaternion((0.5, 0.5, -0.5, -0.5))),
        "TOP":    (mathutils.Vector(( 0.0,  0.0, -1.0)), mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))),
        "BOTTOM": (mathutils.Vector(( 0.0,  0.0,  1.0)), mathutils.Quaternion((0.0, 1.0, 0.0, 0.0))),
    }
    # Open the workspace from the opposite member of Blender's standard
    # front/back pair.  Blender's axis gizmo describes the direction from
    # the specimen toward the viewer, whereas canonical_forward describes
    # the direction from the viewer toward the specimen.
    viewport_forward = -canonical_forward
    viewport_view = max(
        standard_views,
        key=lambda name: viewport_forward.dot(standard_views[name][0]),
    )
    viewport_rotation = standard_views[viewport_view][1]

    # Configure every saved 3D viewport so opening the .blend starts from the
    # canonical preset side with Blender +Z visually upward.  Keep perspective
    # projection for a comfortable working view; only the orientation comes
    # from Blender's standard numpad view.
    view_center = cam_target.copy()
    view_distance = max(diameter * 2.5, 0.001)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            region_3d = space.region_3d
            region_3d.view_perspective = "PERSP"
            region_3d.view_location = view_center
            region_3d.view_distance = view_distance
            region_3d.view_rotation = viewport_rotation

# -------------------------------------------------
# horizontal image orientation
# -------------------------------------------------

# The Fiji-style horizontal flip belongs to the exported scene, never to the
# Blender camera.  Apply it in BOTH upright modes.  Earlier revisions placed
# this block only inside upright_scene=True, which silently removed mirroring
# for the normal Blender export.
if c.get("flip_horizontal", False):
    preset = c.get("preset")
    if preset is None:
        raise RuntimeError(
            "Horizontal camera flipping requires a camera preset."
        )

    preset_view_dirs = {
        "x_front": (-1.0, 0.0, 0.0),
        "x_back":  ( 1.0, 0.0, 0.0),
        "y_front": (0.0, -1.0, 0.0),
        "y_back":  (0.0,  1.0, 0.0),
        "z_front": (0.0, 0.0, -1.0),
        "z_back":  (0.0, 0.0,  1.0),
    }
    preset_up_dirs = {
        "x_front": (0.0, -1.0, 0.0),
        "x_back":  (0.0, -1.0, 0.0),
        "y_front": (0.0, 0.0, -1.0),
        "y_back":  (0.0, 0.0, -1.0),
        "z_front": (0.0, -1.0, 0.0),
        "z_back":  (0.0, -1.0, 0.0),
    }

    canonical_forward = mathutils.Vector(preset_view_dirs[preset])
    canonical_up = mathutils.Vector(preset_up_dirs[preset])
    if settings.get("upright_scene", False):
        canonical_forward = upright_rotation @ canonical_forward
        canonical_up = upright_rotation @ canonical_up
    canonical_forward.normalize()
    canonical_up.normalize()
    canonical_right = canonical_forward.cross(canonical_up).normalized()

    r = canonical_right
    reflection3 = mathutils.Matrix.Identity(3) - 2.0 * mathutils.Matrix((
        (r.x * r.x, r.x * r.y, r.x * r.z),
        (r.y * r.x, r.y * r.y, r.y * r.z),
        (r.z * r.x, r.z * r.y, r.z * r.z),
    ))
    reflection4 = reflection3.to_4x4()
    reflection_about_target = (
        mathutils.Matrix.Translation(target)
        @ reflection4
        @ mathutils.Matrix.Translation(-target)
    )
    for obj in list(bpy.context.scene.objects):
        if obj is not cam:
            obj.matrix_world = reflection_about_target @ obj.matrix_world
    final_center = reflection_about_target @ final_center

# -------------------------------------------------
# final camera-relative lighting
# -------------------------------------------------

# Recompute the basis from the final Blender camera.  This is deliberately
# done after upright_scene and horizontal reflection so placement names keep
# exactly the same screen-space meaning in both upright modes.
final_forward = (target - cam.location).normalized()
final_up = final_camera_up.normalized()
final_right = final_forward.cross(final_up).normalized()
final_up = final_right.cross(final_forward).normalized()

light_data = bpy.data.lights.new(
    "Key",
    type="AREA",
)
light_data.energy = settings["light_power"]
light_data.shape = "SQUARE"
light_data.size = 2.0 * diameter

if settings["light_temperature"] is not None:
    light_data.use_temperature = True
    light_data.temperature = settings["light_temperature"]

light = bpy.data.objects.new("Key", light_data)
bpy.context.collection.objects.link(light)

placement_vectors = {
    "top-left": (-1, 1),
    "top": (0, 1),
    "top-right": (1, 1),
    "left": (-1, 0),
    "center": (0, 0),
    "right": (1, 0),
    "bottom-left": (-1, -1),
    "bottom": (0, -1),
    "bottom-right": (1, -1),
}
px, py = placement_vectors[settings["light_placement"]]
offset = final_right * px + final_up * py
if offset.length > 0:
    offset.normalize()

angle = math.radians(settings["light_angle"])
light_direction = (
    -final_forward * math.cos(angle)
    + offset * math.sin(angle)
).normalized()
light_distance = 1.5 * diameter
light.location = final_center + light_direction * light_distance
light.rotation_euler = (
    (final_center - light.location)
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
scene.view_settings.view_transform = settings["view_transform"]
scene.view_settings.look = settings["look"]
scene.view_settings.exposure = settings["exposure"]
scene.view_settings.gamma = settings["gamma"]

width, height = settings["output_size"]
scene.render.resolution_x = int(width)
scene.render.resolution_y = int(height)
scene.render.resolution_percentage = 100
scene.cycles.samples = int(settings["samples"])

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

        filename = Path(filename).expanduser().resolve()
        filename.parent.mkdir(parents=True, exist_ok=True)

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
            