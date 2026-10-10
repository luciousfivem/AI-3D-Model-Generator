"""Orientation, scaling and bed placement (print frame: +Z up, bed at z = 0, mm units)."""

from __future__ import annotations

import numpy as np
import trimesh

# Rotation that carries the named source "up" axis onto +Z.
_UP_TO_Z = {
    "z": np.eye(3),
    "-z": np.diag([1.0, -1.0, -1.0]),
    "y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
    "-y": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
    "x": np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]),
    "-x": np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
}
UP_AXES = tuple(_UP_TO_Z)


def up_axis_rotation(up: str) -> np.ndarray:
    """3x3 rotation taking the given source up-axis (e.g. ``"y"`` for glTF-style meshes) to +Z."""
    return _UP_TO_Z[up.lower()].copy()


def axis_rotation(axis: str, degrees: float) -> np.ndarray:
    """3x3 rotation about the world X, Y or Z axis."""
    direction = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis.lower()]
    return trimesh.transformations.rotation_matrix(np.radians(degrees), direction)[:3, :3]


def rotate(mesh: trimesh.Trimesh, rotation: np.ndarray) -> trimesh.Trimesh:
    """Rotate about the mesh centre (returns a copy)."""
    out = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    centre = out.bounds.mean(axis=0)
    out.apply_translation(-centre)
    out.apply_transform(transform)
    out.apply_translation(centre)
    return out


def drop_to_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre on X/Y and rest the lowest point on z = 0."""
    out = mesh.copy()
    low, high = out.bounds
    out.apply_translation([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    return out


def scale_to_height(mesh: trimesh.Trimesh, height_mm: float) -> trimesh.Trimesh:
    """Uniformly scale so the Z extent equals ``height_mm``."""
    out = mesh.copy()
    current = float(out.extents[2])
    if current <= 0:
        raise ValueError("Mesh has no height")
    out.apply_scale(height_mm / current)
    return out


def scale_to_largest_dimension(mesh: trimesh.Trimesh, size_mm: float) -> trimesh.Trimesh:
    out = mesh.copy()
    current = float(np.max(out.extents))
    if current <= 0:
        raise ValueError("Mesh has no size")
    out.apply_scale(size_mm / current)
    return out


def _support_mask(
    normals_z: np.ndarray, face_top_z: np.ndarray, bed_z: float, tolerance: float, max_overhang_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """(needs_support, rests_on_bed) for faces given their normal Z and highest corner Z.

    A face rests on the bed only if *all* its corners are at bed level (a flat face lying on the plate);
    a sloped face that merely touches it at a tip does not count.
    """
    on_bed = face_top_z <= bed_z + tolerance
    needs_support = (normals_z < -np.sin(np.radians(max_overhang_deg))) & ~on_bed
    return needs_support, on_bed


def overhang_mask(
    mesh: trimesh.Trimesh,
    max_overhang_deg: float = 45.0,
    rotation: np.ndarray | None = None,
    bed_tolerance_mm: float | None = None,
) -> np.ndarray:
    """Boolean mask of faces that would need support, assuming +Z up and the bed at the lowest point.

    ``max_overhang_deg`` is the steepest overhang printable without support, measured from vertical
    (a vertical wall is 0, a flat ceiling is 90). Faces resting flat on the bed never need support.
    """
    normals = mesh.face_normals
    vertices = np.asarray(mesh.vertices)
    if rotation is not None:
        normals = normals @ rotation.T
        vertices = vertices @ rotation.T
    vertex_z = vertices[:, 2]
    face_top_z = vertex_z[mesh.faces].max(axis=1)
    height = float(vertex_z.max() - vertex_z.min()) if len(vertex_z) else 0.0
    tolerance = bed_tolerance_mm if bed_tolerance_mm is not None else max(1e-6, height * 0.005)
    needs_support, _ = _support_mask(normals[:, 2], face_top_z, float(vertex_z.min()), tolerance, max_overhang_deg)
    return needs_support


def _candidate_rotations(mesh: trimesh.Trimesh, max_facets: int) -> list[np.ndarray]:
    """Orientations worth trying: each axis pointing up, plus the largest flat facets of the hull down."""
    candidates = [np.eye(3)]
    for up in ("-z", "y", "-y", "x", "-x"):
        candidates.append(up_axis_rotation(up))

    try:
        hull = mesh.convex_hull
        order = np.argsort(hull.facets_area)[::-1][:max_facets]
        for index in order:
            normal = hull.facets_normal[index]
            candidates.append(_rotation_taking(normal, np.array([0.0, 0.0, -1.0])))
    except Exception:
        pass  # degenerate hull: the axis-aligned candidates still work
    return candidates


def _rotation_taking(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    transform = trimesh.geometry.align_vectors(source, target)
    return np.asarray(transform)[:3, :3]


def auto_orient(
    mesh: trimesh.Trimesh,
    max_overhang_deg: float = 45.0,
    max_facets: int = 12,
) -> tuple[np.ndarray, dict]:
    """Pick the orientation that needs the least support.

    Returns the 3x3 rotation to apply and a dict with the winning score parts. Candidates are scored by
    the share of surface area that overhangs, with a bonus for a large flat contact patch (adhesion)
    and a small penalty for tall prints.
    """
    total_area = float(mesh.area) or 1.0
    areas = mesh.area_faces
    longest = float(np.max(mesh.extents)) or 1.0

    best = (np.inf, np.eye(3), {})
    vertices = np.asarray(mesh.vertices)
    for rotation in _candidate_rotations(mesh, max_facets):
        vertex_z = (vertices @ rotation.T)[:, 2]
        normals_z = (mesh.face_normals @ rotation.T)[:, 2]
        face_top_z = vertex_z[mesh.faces].max(axis=1)
        height = float(vertex_z.max() - vertex_z.min())
        tolerance = max(1e-6, height * 0.005)
        overhang, on_bed = _support_mask(normals_z, face_top_z, float(vertex_z.min()), tolerance, max_overhang_deg)
        overhang_fraction = float(areas[overhang].sum()) / total_area
        contact_fraction = float(areas[on_bed & (normals_z < -0.95)].sum()) / total_area
        score = overhang_fraction - 0.5 * min(contact_fraction, 0.5) + 0.05 * (height / longest)
        if score < best[0]:
            best = (
                score,
                rotation,
                {
                    "overhang_fraction": overhang_fraction,
                    "contact_fraction": contact_fraction,
                    "height_mm": height,
                },
            )
    return best[1], best[2]
