
import numpy as np
import pyvista as pv

from squirrel.library.render.scene import make_colormap


class PyVistaRenderer:

    def __init__(
        self,
        off_screen=False,
        image_size=(1000, 1000),
        world_scale=1.0,
    ):
        self.off_screen = off_screen
        self.image_size = image_size
        self.world_scale = world_scale

        self.plotter = None

    def _create_plotter(self, window_size):

        self.plotter = pv.Plotter(
            off_screen=self.off_screen,
            window_size=window_size,
        )

        return self.plotter

    # Output
    def show(self, scene):

        self._create_plotter(self.image_size)
        self._load_scene(scene)

        self.plotter.show(auto_close=False)

        camera = self.plotter.camera_position
        camera_position = [
            tuple(np.asarray(camera[0]) / self.world_scale),
            tuple(np.asarray(camera[1]) / self.world_scale),
            tuple(camera[2]),
        ]
        print("\nCamera position:")
        print(camera_position)

        self.plotter.close()
        self.plotter = None

    def screenshot(self, scene, filename):

        from PIL import Image

        self._create_plotter(self.image_size)
        self._load_scene(scene)

        image = self.plotter.screenshot(
            return_img=True,
            window_size=self.image_size,
        )

        self.plotter.close()
        self.plotter = None

        if scene.camera_flip_horizontal:
            image = image[:, ::-1]

        Image.fromarray(image).save(filename)

    # Internal helpers
    def _apply_colors(self, obj):

        cmap = make_colormap(
            base_color=obj.color,
            hue_shift=np.abs(obj.hue_shift),
            lightness_shift=obj.lightness_shift,
            saturation_shift=0,
        )

        # Keep mesh and camera in the same (scaled) coordinate system.
        mesh = obj.mesh.copy()
        mesh.points *= self.world_scale
        self._update_view_normals(mesh, obj)

        self.plotter.add_mesh(
            mesh,
            scalars="view_normal",
            cmap=cmap,
            smooth_shading=True,
            ambient=0.2,
            diffuse=0.8,
            specular=0.2,
            specular_power=20,
            show_scalar_bar=False,
        )

    def _update_view_normals(self, mesh, obj):

        normals = mesh.point_normals
        points = mesh.points

        camera = np.asarray(self.plotter.camera.position, dtype=float)
        view_up = np.asarray(self.plotter.camera.up, dtype=float)

        view_dir = camera - points
        view_dir /= np.linalg.norm(
            view_dir,
            axis=1,
            keepdims=True,
        )

        if obj.normal_mode == "front_silhouette":

            nv = np.abs(
                np.sum(normals * view_dir, axis=1)
            )
            view_normal = 1.0 - 2.0 * nv

        elif obj.normal_mode == "left_right":

            focal_point = np.asarray(
                self.plotter.camera.focal_point,
                dtype=float,
            )
            camera_dir = focal_point - camera
            camera_dir /= np.linalg.norm(camera_dir)

            right = np.cross(camera_dir, view_up)
            right /= np.linalg.norm(right)

            nv = normals @ right
            rim_strength = 2.0

            view_normal = np.sign(nv) * (
                np.abs(nv) ** rim_strength
            )

        else:
            raise ValueError(
                f"Invalid normal mode: {obj.normal_mode}"
            )

        mesh["view_normal"] = (
            np.sign(obj.hue_shift)
            * np.clip(view_normal, -1, 1)
        )

    def _scale_camera(self, camera_position):

        if camera_position is None:
            return None

        return [
            tuple(np.array(camera_position[0]) * self.world_scale),
            tuple(np.array(camera_position[1]) * self.world_scale),
            camera_position[2],
        ]

    def _add_slice(self, em_slice):

        volume = em_slice.volume
        data = volume.data

        sz, sy, sx = (
            np.asarray(volume.voxel_size, dtype=float)
            * self.world_scale
        )
        oz, oy, ox = (
            np.asarray(volume.origin, dtype=float)
            * self.world_scale
        )

        axis = em_slice.axis
        index = em_slice.index
        nz, ny, nx = data.shape

        # Data stays (z, y, x); geometry is created in world (x, y, z).
        if axis == "z":
            image = data[index, :, :]
            yy, xx = np.meshgrid(
                oy + np.arange(ny) * sy,
                ox + np.arange(nx) * sx,
                indexing="ij",
            )
            zz = np.full_like(xx, oz + index * sz)
            grid = pv.StructuredGrid(xx, yy, zz)

        elif axis == "y":
            image = data[:, index, :]
            zz, xx = np.meshgrid(
                oz + np.arange(nz) * sz,
                ox + np.arange(nx) * sx,
                indexing="ij",
            )
            yy = np.full_like(xx, oy + index * sy)
            grid = pv.StructuredGrid(xx, yy, zz)

        elif axis == "x":
            image = data[:, :, index]
            zz, yy = np.meshgrid(
                oz + np.arange(nz) * sz,
                oy + np.arange(ny) * sy,
                indexing="ij",
            )
            xx = np.full_like(yy, ox + index * sx)
            grid = pv.StructuredGrid(xx, yy, zz)

        else:
            raise ValueError(f"Invalid slice axis: {axis}")

        grid.point_data["em"] = image.ravel(order="F")

        self.plotter.add_mesh(
            grid,
            scalars="em",
            cmap=em_slice.cmap,
            clim=em_slice.clim,
            opacity=em_slice.opacity,
            show_scalar_bar=False,
            lighting=False,
        )

    def _load_scene(self, scene):

        self.plotter.set_background(scene.background)

        # Set the final camera before computing camera-dependent colors.
        if scene.camera_position is not None:
            self.plotter.camera_position = (
                self._scale_camera(scene.camera_position)
            )

        for obj in scene.objects:
            self._apply_colors(obj)

        for em_slice in scene.slices:
            self._add_slice(em_slice)

        # Canonical presets are fitted to the complete underlying data extent.
        # Perspective fitting uses all 8 corners, so depth as well as width and
        # height is accounted for.
        if scene.camera_preset is not None:
            bounds_min, bounds_max = scene.get_bounds()
            bounds_min = np.asarray(bounds_min, dtype=float) * self.world_scale
            bounds_max = np.asarray(bounds_max, dtype=float) * self.world_scale

            if scene.camera_projection == "orthographic":
                extent = bounds_max - bounds_min
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

                aspect = self.image_size[0] / self.image_size[1]
                required_height = max(height, width / aspect)

                self.plotter.camera.parallel_projection = True
                self.plotter.camera.parallel_scale = (
                    0.5
                    * required_height
                    * (1.0 + scene.camera_fit_padding)
                )

            elif scene.camera_projection == "perspective":
                self.plotter.camera.parallel_projection = False

                camera = self.plotter.camera
                focal = np.asarray(camera.focal_point, dtype=float)
                position = np.asarray(camera.position, dtype=float)
                up = np.asarray(camera.up, dtype=float)

                camera_side = position - focal
                camera_side /= np.linalg.norm(camera_side)

                forward = -camera_side
                up /= np.linalg.norm(up)
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                up = np.cross(right, forward)
                up /= np.linalg.norm(up)

                corners = np.array([
                    [x, y, z]
                    for x in (bounds_min[0], bounds_max[0])
                    for y in (bounds_min[1], bounds_max[1])
                    for z in (bounds_min[2], bounds_max[2])
                ])
                offsets = corners - focal

                aspect = self.image_size[0] / self.image_size[1]
                tan_v = np.tan(np.deg2rad(camera.view_angle) * 0.5)
                tan_h = tan_v * aspect
                padding = 1.0 + scene.camera_fit_padding

                along = offsets @ camera_side
                horizontal = np.abs(offsets @ right)
                vertical = np.abs(offsets @ up)

                required_distance = np.max(
                    along
                    + np.maximum(
                        padding * horizontal / tan_h,
                        padding * vertical / tan_v,
                    )
                )

                camera.position = tuple(
                    focal + camera_side * required_distance
                )
                self.plotter.reset_camera_clipping_range()

            else:
                raise ValueError(
                    f"Invalid camera projection: {scene.camera_projection}"
                )
