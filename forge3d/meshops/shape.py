"""Shape-changing operations: decimation, smoothing and flat-base cutting."""

from __future__ import annotations

import numpy as np
import trimesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse down to roughly ``target_faces`` triangles."""
    if target_faces >= len(mesh.faces):
        return mesh
    import fast_simplification

    points, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=1.0 - target_faces / len(mesh.faces),
    )
    return trimesh.Trimesh(vertices=points, faces=faces, process=True)


def smooth(mesh: trimesh.Trimesh, passes: int) -> trimesh.Trimesh:
    """Taubin smoothing: removes marching-cubes stair-stepping without shrinking the model.

    One pass is a shrink step followed by a dilate step. trimesh takes ``nu`` as a positive number and
    applies it with the opposite sign; 1/lamb - 1/nu must lie in (0, 0.1) for the filter to be stable.
    """
    if passes <= 0:
        return mesh
    out = mesh.copy()
    trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=0.52, iterations=2 * int(passes))
    return out


def cut_flat_base(mesh: trimesh.Trimesh, depth_mm: float) -> trimesh.Trimesh:
    """Slice ``depth_mm`` off the bottom and cap it, giving a flat face for bed adhesion.

    Expects the mesh resting on z = 0. Returns the input unchanged if the cut would remove (nearly)
    everything or the cap cannot be built.
    """
    if depth_mm <= 0:
        return mesh
    low = float(mesh.bounds[0][2])
    if depth_mm >= float(mesh.extents[2]) * 0.5:
        return mesh
    try:
        cut = mesh.slice_plane(
            plane_origin=[0.0, 0.0, low + depth_mm], plane_normal=[0.0, 0.0, 1.0], cap=True
        )
    except Exception:
        return mesh
    if cut is None or len(cut.faces) == 0:
        return mesh
    return cut
