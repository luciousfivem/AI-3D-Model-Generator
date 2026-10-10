"""Instant, no-AI image -> 3D: inflate a picture's silhouette into a rounded "pillow".

Not a replacement for the neural engines: it knows nothing about the back or the depth of the subject.
It is useful for logos, icons, badges and cut-out shapes, works on any computer in about a second, and
gives the whole app (viewport, print prep, export) something to run on before the AI models are set up.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from .base import CancelToken, EngineInfo, ImageTo3DEngine, ProgressFn

_RESOLUTION = {"draft": 96, "standard": 160, "high": 224, "ultra": 288}


class InflateEngine(ImageTo3DEngine):
    native_up = "y"  # picture "up" is +Y; thickness runs along Z
    info = EngineInfo(
        key="inflate",
        label="Inflate silhouette - instant, no AI",
        description="Puffs the outline of the picture into a 3D pillow. Best for logos, icons and flat shapes.",
        license="MIT (this app)",
        commercial_use=True,
        approx_vram_gb=0.0,
        download_gb=0.0,
    )

    def check_available(self) -> None:
        pass

    def load(self, progress: ProgressFn, cancel: CancelToken) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def loaded(self) -> bool:
        return True

    def generate(
        self,
        image: Image.Image,
        *,
        quality: str,
        seed: int,
        progress: ProgressFn,
        cancel: CancelToken,
    ) -> trimesh.Trimesh:
        cancel.check()
        side = _RESOLUTION.get(quality, 160)
        rgba = image.convert("RGBA")
        rgba.thumbnail((side, side))
        mask = np.asarray(rgba.getchannel("A")) > 128
        if mask.sum() < 16:
            # no transparency to work with: treat everything that is not near-white as the subject
            grey = np.asarray(rgba.convert("L"))
            mask = grey < 235
        mask = ndimage.binary_fill_holes(np.pad(mask, 3))
        if mask.sum() < 16:
            raise ValueError("Could not find a subject in the picture to inflate.")

        progress(0.2, "Inflating silhouette...")
        distance = ndimage.distance_transform_edt(mask)
        # half-thickness grows with the square root of the distance from the edge: round, not conical
        half = np.sqrt(distance / distance.max()) * (0.35 * max(mask.shape))
        half = ndimage.gaussian_filter(half, 1.5) * (half > 0)  # round off the discrete steps of the distance map
        layers = int(np.ceil(half.max())) * 2 + 8
        z = np.arange(layers) - layers / 2 + 0.5
        volume = (np.abs(z)[:, None, None] < half[None, :, :]).astype(np.float32)
        volume = ndimage.gaussian_filter(volume, 1.0)
        cancel.check()

        from skimage import measure

        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
        # volume axes are (z, row, col) -> (x = col, y = up, z = depth)
        rows = mask.shape[0]
        mesh = trimesh.Trimesh(
            np.column_stack([verts[:, 2], rows - verts[:, 1], verts[:, 0] - layers / 2]),
            faces[:, ::-1],  # flipping the row axis mirrors the surface, so flip the winding back
            process=True,
        )
        trimesh.repair.fix_normals(mesh)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.52, iterations=16)  # volume-preserving
        progress(0.95, "Shape generated")
        return mesh
