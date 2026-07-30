
import numpy as np
import os
import hashlib
import json


class ColorManager:
    """Assign colors to objects."""

    def __init__(
        self,
        palette,
        mode="sequential",
        seed=None,
        exclude_neutral=False,
        hue_shift=0.05,
        lightness_shift=0.18,
        normal_mode="front_silhouette",
    ):

        self.palette = palette
        self.mode = mode
        self.seed = seed

        self.exclude_neutral = exclude_neutral

        self.hue_shift = hue_shift
        self.lightness_shift = lightness_shift
        self.normal_mode = normal_mode

        self.palette = self._load_palette(palette)

    def _load_palette(self, palette):
        import tol_colors
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        try:
            palette = getattr(tol_colors, palette)
        except AttributeError:

            available = sorted(
                name for name in dir(tol_colors)
                if not name.startswith("_")
            )

            raise ValueError(
                f"Unknown Tol palette '{palette}'.\n"
                f"Available palettes:\n"
                + "\n".join(available)
            )

        # Already a colormap -> return as-is
        if isinstance(
            palette,
            (ListedColormap, LinearSegmentedColormap),
        ):
            return palette

        # Factory function -> instantiate
        if callable(palette):
            return palette()

        return palette

    # def _load_palette(self, palette):
    #     import tol_colors

    #     try:
    #         palette_fn = getattr(tol_colors, palette)
    #     except AttributeError:

    #         available = sorted(
    #             name for name in dir(tol_colors)
    #             if not name.startswith("_")
    #         )

    #         raise ValueError(
    #             f"Unknown Tol palette '{palette}'.\n"
    #             f"Available palettes:\n"
    #             + "\n".join(available)
    #         )

    #     if not callable(palette_fn):
    #         raise TypeError(f"{palette} is not a palette.")

    #     return palette_fn()

    def _is_discrete(self):

        return isinstance(self.palette, (list, tuple))

    def _is_colormap(self):
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        return isinstance(
            self.palette,
            (ListedColormap, LinearSegmentedColormap),
        )

    def _assign_discrete(self, objects):

        colors = list(self.palette)

        if self.exclude_neutral:
            colors = [
                c for c in colors
                if max(c[:3]) - min(c[:3]) > 0.05
            ]

        if len(colors) == 0:
            raise ValueError("No colors left after excluding neutral colors.")

        rng = np.random.default_rng(self.seed)

        n_colors = len(colors)
        cycle_colors = colors.copy()

        for idx, obj in enumerate(objects):

            pos = idx % n_colors

            if self.mode == "random" and pos == 0:
                cycle_colors = colors.copy()
                rng.shuffle(cycle_colors)

            color = cycle_colors[pos]

            obj.set_color(
                color=color,
                hue_shift=self.hue_shift,
                lightness_shift=self.lightness_shift,
                normal_mode=self.normal_mode,
            )

    def _assign_continuous(self, objects):

        n = len(objects)

        if n == 1:
            values = np.array([0.5])
        else:
            values = np.linspace(0, 1, n)

        if self.mode == "random":
            rng = np.random.default_rng(self.seed)
            rng.shuffle(values)

        for obj, value in zip(objects, values):

            color = self.palette(value)

            obj.set_color(
                color=color,
                hue_shift=self.hue_shift,
                lightness_shift=self.lightness_shift,
                normal_mode=self.normal_mode,
            )

    def assign(self, objects):

        if self._is_discrete():
            self._assign_discrete(objects)

        elif self._is_colormap():
            self._assign_continuous(objects)

        else:
            raise TypeError(
                f"Unsupported palette type: {type(self.palette)}"
            )

        return objects


def make_colormap(
    base_color="tomato",
    hue_shift=0.05,
    lightness_shift=0.18,
    saturation_shift=0.10,
    N=256,
):

    import colorsys
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    r, g, b = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    colors = []

    for t in np.linspace(-1, 1, N):
        hh = (h + hue_shift * t) % 1.0
        ll = np.clip(l + lightness_shift * t, 0, 1)
        ss = np.clip(s - saturation_shift * abs(t), 0, 1)

        colors.append(colorsys.hls_to_rgb(hh, ll, ss))

    return LinearSegmentedColormap.from_list("custom", colors)  


class SegmentObject:
    """Represents a single labeled object in a segmentation."""

    def __init__(self, label, mask=None, voxel_size=(1, 1, 1), offset=(0, 0, 0)):
        self.label = label
        self.mask = mask
        self.voxel_size = voxel_size

        self.raw_mesh = None
        self.mesh = None
        self.offset = offset

        # Appearance
        self.name = None
        self.color = "tomato"
        self.hue_shift = 0.05
        self.lightness_shift = 0.18
        self.normal_mode = "front_silhouette"

    @staticmethod
    def _gaussian_padding(sigma, factor=3):
        if sigma is None:
            return (1, 1, 1)

        if np.isscalar(sigma):
            sigma = (sigma, sigma, sigma)
            
        return tuple(
            max(1, int(np.ceil(factor * s)))
            for s in sigma
        )

    def to_mesh(
        self,
        level=0.5,
        gaussian_sigma=None,
        cache_dir=None,
        force=False,
    ):
        """Extract a surface mesh from the object's mask."""
        import pyvista as pv

        cache_file = self.cache_file(
            cache_dir,
            "mesh",
            level=level,
            gaussian_sigma=gaussian_sigma,
        )

        if cache_file and os.path.exists(cache_file) and not force:
            self.raw_mesh = pv.read(cache_file)
            self.mesh = self.raw_mesh.copy()
            return self

        # This lets the object appear filled
        pad = self._gaussian_padding(gaussian_sigma)

        segmentation = np.pad(
            self.mask.astype(np.float32),
            tuple((p, p) for p in pad),
            mode="constant",
        )

        if gaussian_sigma is not None:
            from vigra.filters import gaussianSmoothing
            segmentation = gaussianSmoothing(
                segmentation, sigma=gaussian_sigma
            )

        grid = pv.ImageData()
        grid.dimensions = segmentation.shape
        grid.spacing = self.voxel_size

        grid.point_data["labels"] = segmentation.astype(np.float32).ravel(order="F")

        self.raw_mesh = grid.contour(
            isosurfaces=[level],
            scalars="labels",
        )

        oz, oy, ox = self.offset
        self.raw_mesh.points += np.array([
            (oz - pad[0]) * self.voxel_size[0],
            (oy - pad[1]) * self.voxel_size[1],
            (ox - pad[2]) * self.voxel_size[2],
        ])

        self.mesh = self.raw_mesh.copy()

        self.mesh = self.mesh.compute_normals(
            auto_orient_normals=True,
            inplace=False,
        )

        if cache_file:
            self.raw_mesh.save(cache_file)

        return self

    def smooth(
        self,
        stage="smooth",
        cache_dir=None,
        force=False,
        smooth_iterations=100,
        smooth_relaxation=0.01,
        smooth_xy_only=False,
    ):
        """Smooth the current mesh."""
        import pyvista as pv

        cache_file = self.cache_file(
            cache_dir,
            stage,
            smooth_iterations=smooth_iterations,
            smooth_relaxation=smooth_relaxation,
            smooth_xy_only=smooth_xy_only,
        )

        if cache_file and os.path.exists(cache_file) and not force:
            self.mesh = pv.read(cache_file)
            return self

        print(f'smooting with cache_file = {cache_file}')

        if self.raw_mesh is None:
            raise RuntimeError("Generate mesh first.")

        self.mesh = self.raw_mesh.copy()

        if smooth_iterations <= 0:
            return self

        if not smooth_xy_only:
            self.mesh = self.mesh.smooth(
                n_iter=smooth_iterations,
                relaxation_factor=smooth_relaxation,
            )
            if cache_file:
                self.mesh.save(cache_file)
            return self

        original = self.mesh.points.copy()

        smoothed = self.mesh.copy().smooth(
            n_iter=smooth_iterations,
            relaxation_factor=smooth_relaxation,
        )

        displacement = smoothed.points - original
        displacement[:, 0] = 0  # Preserve x coordinate

        self.mesh = self.mesh.copy()
        self.mesh.points = original + displacement

        self.mesh = self.mesh.compute_normals(
            auto_orient_normals=True,
            inplace=False,
        )

        if cache_file:
            self.mesh.save(cache_file)

        return self

    def decimate(
        self,
        target_reduction=0.5,
    ):
        """Reduce the number of triangles."""

        self.mesh = self.mesh.decimate(
            target_reduction=target_reduction,
        )

        self.mesh = self.mesh.compute_normals(
            auto_orient_normals=True,
            inplace=False,
        )

        return self

    def set_color(
        self,
        color="tomato",
        hue_shift=0.05,
        lightness_shift=0.18,
        normal_mode="front_silhouette",
    ):
        """Store coloring settings for the object."""

        self.color = color
        self.hue_shift = hue_shift
        self.lightness_shift = lightness_shift
        self.normal_mode = normal_mode

        return self

    def clean(self):
        pass

    def save_mesh(self, filename):
        """Save cached mesh to disk."""

        if self.mesh is None:
            raise RuntimeError("No mesh available.")

        if not os.path.exists(os.path.split(filename)[0]):
            os.mkdir(os.path.split(filename)[0])

        self.mesh.save(filename)

        return self

    def load_mesh(self, filename):
        import pyvista as pv
        """Load mesh from disk."""

        self.mesh = pv.read(filename)

        return self

    def cache_key(self, stage, **kwargs):
        """Generate a unique cache key for this object and processing stage."""

        payload = {
            "label": int(self.label),
            "offset": tuple(int(v) for v in self.offset),
            "shape": tuple(int(v) for v in self.mask.shape),
            "stage": stage,
            **kwargs,
        }

        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    def cache_file(self, cache_dir, stage, **kwargs):

        if cache_dir is None:
            return None

        os.makedirs(cache_dir, exist_ok=True)

        key = self.cache_key(stage, **kwargs)

        return os.path.join(
            cache_dir,
            f"{key}.vtp",
        )
    

class Segmentation:
    """Container for a labeled segmentation."""

    def __init__(self, data=None, voxel_size=(1, 1, 1)):
        self.data = data
        self.voxel_size = voxel_size
        self.objects = []

    def load(self):
        pass

    def extract_objects(
        self,
        cache_dir=None,
        force=False,
    ):

        cache_file = self.cache_file(cache_dir)

        # --------------------------------------------------
        # Load cache

        if cache_file and os.path.exists(cache_file) and not force:

            print('    loading cache ...')

            data = np.load(cache_file, allow_pickle=True)

            self.objects = [
                SegmentObject(
                    label=int(label),
                    mask=mask,
                    voxel_size=self.voxel_size,
                    offset=tuple(offset),
                )
                for label, mask, offset in zip(
                    data["labels"],
                    data["masks"],
                    data["offsets"],
                )
            ]

            print('    returning data ...')
            return self

        # --------------------------------------------------
        # Compute objects

        self.objects = []

        for label in np.unique(self.data):

            if label == 0:
                continue

            mask = self.data == label

            coords = np.argwhere(mask)

            z0, y0, x0 = coords.min(axis=0) 
            z1, y1, x1 = coords.max(axis=0) + 1

            self.objects.append(
                SegmentObject(
                    label=int(label),
                    mask=mask[z0:z1, y0:y1, x0:x1],
                    voxel_size=self.voxel_size,
                    offset=(z0, y0, x0),
                )
            )

        # --------------------------------------------------
        # Save cache

        if cache_file:

            masks = np.empty(len(self.objects), dtype=object)
            for i, obj in enumerate(self.objects):
                masks[i] = obj.mask

            offsets = np.empty(len(self.objects), dtype=object)
            for i, obj in enumerate(self.objects):
                offsets[i] = obj.offset

            np.savez_compressed(
                cache_file,
                labels=np.array([o.label for o in self.objects], dtype=np.int64),
                masks=masks,
                offsets=np.array([o.offset for o in self.objects], dtype=np.int64),
            )

        return self

    def get_object(self, label):
        """Return the object with the given label."""

        for obj in self.objects:
            if obj.label == label:
                return obj

        raise KeyError(f"Label {label} not found.")

    def __iter__(self):
        return iter(self.objects)

    def process_objects(
        self,
        cache_dir=None,
        to_mesh_kwargs=None,
        smooth_xy_kwargs=None,
        smooth_xyz_kwargs=None,
        color_kwargs=None,
    ):

        to_mesh_kwargs = to_mesh_kwargs or {}

        for obj in self.objects:

            obj.to_mesh(
                cache_dir=cache_dir,
                **to_mesh_kwargs,
            )

            if smooth_xy_kwargs is not None:
                obj.smooth(
                    stage="smooth_xy",
                    cache_dir=cache_dir,
                    **smooth_xy_kwargs,
                )

            if smooth_xyz_kwargs is not None:
                obj.smooth(
                    stage="smooth_xyz",
                    cache_dir=cache_dir,
                    **smooth_xyz_kwargs,
                )

            if color_kwargs is not None:
                obj.set_color(**color_kwargs)

        return self
    
    def assign_colors(
        self,
        palette,
        mode="sequential",
        seed=None,
        hue_shift=0.05,
        lightness_shift=0.18,
        normal_mode="front_silhouette",
        exclude_neutral=False
    ):

        manager = ColorManager(
            palette=palette,
            mode=mode,
            hue_shift=hue_shift,
            lightness_shift=lightness_shift,
            normal_mode=normal_mode,
            seed=seed,
            exclude_neutral=exclude_neutral
        )

        manager.assign(self.objects)

        return self

    def cache_key(self):
        payload = (
            self.data.view(np.uint8).tobytes(),
            str(self.voxel_size).encode(),
        )
        h = hashlib.sha1()
        for p in payload:
            h.update(p)
        return h.hexdigest()[:16]

    def cache_file(self, cache_dir):

        if cache_dir is None:
            return None

        os.makedirs(cache_dir, exist_ok=True)

        return os.path.join(
            cache_dir,
            f"{self.cache_key()}_objects.npz",
        )

    
class EMVolume:
    """Container for EM data."""

    def __init__(self):
        self.data = None
        self.voxel_size = None

    def load(self):
        pass

    def get_slice(self):
        pass


class Scene:

    def __init__(self):
        self.objects = []

        self.background = "black"
        self.camera_position = None
        self.anti_aliasing = True

    # Add data
    def add_object(self, obj):
        """Add a SegmentObject to the scene."""

        self.objects.append(obj)
        return self

    def add_segmentation(self, segmentation):
        for obj in segmentation:
            self.add_object(obj)
        return self

    def add_slice(self):
        pass

    # Appearance
    def update_colors(self):
        pass

    def set_background(self, color):

        self.background = color

        return self

    def set_camera(self, camera_position):

        self.camera_position = camera_position

        return self

    def set_anti_aliasing(self, enabled=True):

        self.anti_aliasing = enabled

        return self


if __name__ == '__main__':

    import os
    from h5py import File

    # ------------------------------------------------------------------
    # Load segmentation

    out_dir = "/media/julian/Data/tmp/pyvista_new"
    os.makedirs(out_dir, exist_ok=True)

    with File(
        "/media/julian/Data/projects/hennies/amst2-publication/segment_crystals/02_pre_alignment_p456_z76_180_uint8_crystals.h5",
        "r",
    ) as f:
        seg = f["data"][:]

    segmentation = (
        Segmentation(
            data=seg,
            voxel_size=(50, 5, 5),
        )
        .extract_objects()
    )

    # ------------------------------------------------------------------
    # Create objects and scene

    scene = Scene()

    # # Manual version
    # for idx, obj in enumerate(segmentation):
    #     (
    #         obj
    #         .to_mesh(cache_file=os.path.join(out_dir, 'cache', 'obj_{:02d}.vtp'.format(idx)))
    #         .smooth(
    #             smooth_iterations=1000,
    #             smooth_xy_only=True, 
    #             cache_file=os.path.join(out_dir, 'cache', 'obj_{:02d}_smooth_xy.vtp'.format(idx))
    #         )
    #         .smooth(
    #             smooth_iterations=100,
    #             smooth_xy_only=False,
    #             cache_file=os.path.join(out_dir, 'cache', 'obj_{:02d}_smooth_xyz.vtp'.format(idx))
    #         )
    #         .set_color(
    #             color='#882255',
    #             hue_shift=0.05,
    #             lightness_shift=0.0,
    #             normal_mode='left_right'
    #         )
    #     )
    #     scene.add_object(obj)
    #     if idx == 5:
    #         break

    # # Internal version
    segmentation = (
        Segmentation(seg, voxel_size=(50, 5, 5))
        .extract_objects(os.path.join(out_dir, "cache"))
        .process_objects(
            cache_dir=os.path.join(out_dir, "cache"),
            to_mesh_kwargs={
                "gaussian_sigma": (0, 1.3, 1.3)
            },
            # smooth_xy_kwargs={
            #     "smooth_iterations": 1000,
            #     "smooth_relaxation": 0.01,
            #     "smooth_xy_only": True,
            # },
            # smooth_xyz_kwargs={
            #     "smooth_iterations": 500,
            #     "smooth_relaxation": 0.01,
            #     "smooth_xy_only": False,
            # },
            # decimate_kwargs={
            #     "target_reduction": 0.9
            # },
            # color_kwargs={
            #     "color": "#882255",
            #     "hue_shift": 0.05,
            #     "normal_mode": "left_right",
            # },
        )
        .assign_colors(
            palette='rainbow_discrete',
            mode='random',
            hue_shift=0.08,
            lightness_shift=0.0,
            normal_mode='left_right',
            seed=4,
            exclude_neutral=True
        )
    )
    scene.add_segmentation(segmentation)

    scene.set_background("white")
    scene.set_anti_aliasing()

    scene.set_camera([
        (2900, 12800, 1300),
        (2000, 2600, 2600),
        (0, 0, 1),
    ])

    # # Pyvista rendering
    # from squirrel.library.render.pyvista_renderer import PyVistaRenderer
    # renderer = PyVistaRenderer(off_screen=True, image_size=(3000, 3000), world_scale=0.001)
    # renderer.screenshot(scene, os.path.join(out_dir, 'scene.png'))
    # # renderer.show(scene)

    # Blender rendering
    from squirrel.library.render.blender_renderer import BlenderRenderer
    renderer = BlenderRenderer(
        samples=128,
        output_size=(3000, 3000),
        world_scale=0.001
    )
    renderer.screenshot(scene, os.path.join(out_dir, 'scene_blender.png'))
    # renderer.write_blend(scene, os.path.join(out_dir, 'scene.blend'))
