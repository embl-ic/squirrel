
import numpy as np
import pyvista as pv

from squirrel.library.render.scene import make_colormap


class PyVistaRenderer:

    def __init__(
        self,
        off_screen=False,
        image_size=(800, 800),
        world_scale=1.0
    ):

        self.plotter = pv.Plotter(
            off_screen=off_screen,
            window_size=image_size,
        )

        self.world_scale = world_scale

    # Output
    def show(self, scene):

        self._load_scene(scene)

        # # This is too expensive
        # self.plotter.camera.AddObserver(
        #     "ModifiedEvent",
        #     lambda *_: self._camera_callback(scene)
        # )

        def print_camera():
            cam = self.plotter.camera
            print([
                tuple(cam.position),
                tuple(cam.focal_point),
                tuple(cam.up),
            ])

        self.plotter.add_key_event("c", print_camera)

        self.plotter.show()

    def screenshot(self, scene, filename):

        self._load_scene(scene)

        print(len(self.plotter.actors))

        self.plotter.show(
            screenshot=filename,
            auto_close=True,
        )

    # Internal helpers
    def _apply_colors(self, obj):

        cmap = make_colormap(
            base_color=obj.color,
            hue_shift=np.abs(obj.hue_shift),
            lightness_shift=obj.lightness_shift,
            saturation_shift=0,
        )

        self._update_view_normals(obj)

        # self.plotter.add_mesh(
        #     obj.mesh,
        #     scalars="view_normal",
        #     cmap=cmap,
        #     smooth_shading=True,
        #     pbr=True,
        #     metallic=0.1,
        #     roughness=0.9,
        #     ambient=0.5,
        #     diffuse=0.5,
        #     show_scalar_bar=False,
        # )

        mesh = obj.mesh.copy()
        mesh.points *= self.world_scale

        self.plotter.add_mesh(
            mesh,
            scalars="view_normal",
            cmap=cmap,
            smooth_shading=True,
            ambient=0.2,
            diffuse=0.8,
            specular=0.2,
            specular_power=20,
            show_scalar_bar=False
        )

    def _update_view_normals(self, obj):

        mesh = obj.mesh

        normals = mesh.point_normals
        points = mesh.points

        camera = np.asarray(self.plotter.camera.position)
        view_up = np.asarray(self.plotter.camera.up)

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

            right = np.cross(view_dir[0], view_up)
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

    def _camera_callback(
        self,
        scene,
    ):

        for obj in scene.objects:

            self._update_view_normals(obj)

        self.plotter.render()

    def _scale_camera(self, camera_position):

        if camera_position is None:
            return None

        return [
            tuple(np.array(camera_position[0]) * self.world_scale),
            tuple(np.array(camera_position[1]) * self.world_scale),
            camera_position[2],
        ]

    def _load_scene(self, scene):

        self.plotter.set_background(
            scene.background
        )

        for obj in scene.objects:

            self._apply_colors(obj)

        if scene.anti_aliasing:
            self.plotter.enable_anti_aliasing()

        if scene.camera_position is not None:
            self.plotter.camera_position = (
                self._scale_camera(scene.camera_position)
            )   
