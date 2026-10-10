"""Mesh cleanup and repair: the steps that turn an AI mesh into something a slicer accepts."""

from __future__ import annotations

import numpy as np
import trimesh


def clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge duplicate vertices and drop NaNs, degenerate/duplicate faces and unused vertices."""
    out = mesh.copy()
    out.remove_infinite_values()
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    return out


def remove_floaters(mesh: trimesh.Trimesh, min_ratio: float = 0.02) -> tuple[trimesh.Trimesh, int]:
    """Drop disconnected pieces smaller than ``min_ratio`` of the largest piece (by face count).

    Generative models often leave tiny detached blobs ("floaters") that would print as loose crumbs.
    Returns the cleaned mesh and the number of pieces removed.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    sizes = np.array([len(p.faces) for p in parts])
    keep = sizes >= max(1, int(sizes.max() * min_ratio))
    removed = int((~keep).sum())
    if removed == 0:
        return mesh, 0
    kept = [p for p, k in zip(parts, keep) if k]
    return (kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)), removed


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Boundary loops as vertex-index lists, following the winding of the existing faces."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = int(directed.max()) + 1
    undirected_key = np.sort(directed, axis=1)
    undirected_key = undirected_key[:, 0] * n + undirected_key[:, 1]
    _, inverse, counts = np.unique(undirected_key, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.reshape(-1)] == 1]
    if len(boundary) == 0:
        return []

    successors: dict[int, list[int]] = {}
    for a, b in boundary.tolist():
        successors.setdefault(a, []).append(b)

    loops: list[list[int]] = []
    used: set[int] = set()
    for start in list(successors):
        if start in used:
            continue
        loop = [start]
        used.add(start)
        current = start
        while True:
            options = successors.get(current)
            if not options:
                break  # dangling edge chain (non-manifold); leave it to the fallback
            nxt = options.pop()
            if nxt == start:
                loops.append(loop)
                break
            if nxt in used:
                break
            loop.append(nxt)
            used.add(nxt)
            current = nxt
    return loops


def fill_holes(mesh: trimesh.Trimesh, max_loop_vertices: int = 5000) -> tuple[trimesh.Trimesh, int]:
    """Close open boundary loops with a centroid fan. Returns the mesh and the number of holes filled.

    A fan is exact for flat holes (e.g. a cut-off base) and merely watertight for curved ones, which is
    what a slicer needs. Loops longer than ``max_loop_vertices`` are left alone.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()  # boundaries are only detectable when coincident vertices are shared
    loops = [lp for lp in _boundary_loops(mesh) if 3 <= len(lp) <= max_loop_vertices]
    if not loops:
        return mesh, 0

    vertices = [np.asarray(mesh.vertices, dtype=np.float64)]
    new_faces = []
    next_index = len(mesh.vertices)
    for loop in loops:
        ring = loop[::-1]  # opposite direction to the existing faces keeps the winding consistent
        if len(ring) == 3:
            new_faces.append(ring)
            continue
        centroid = np.asarray(mesh.vertices)[ring].mean(axis=0)
        vertices.append(centroid[None, :])
        for i in range(len(ring)):
            new_faces.append([next_index, ring[i], ring[(i + 1) % len(ring)]])
        next_index += 1

    out = trimesh.Trimesh(
        vertices=np.concatenate(vertices),
        faces=np.concatenate([mesh.faces, np.asarray(new_faces, dtype=np.int64)]),
        process=False,
    )
    return out, len(loops)


def fix_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make face winding consistent and make sure normals point outward."""
    out = mesh.copy()
    trimesh.repair.fix_winding(out)
    trimesh.repair.fix_inversion(out)
    return out


def voxel_remesh(mesh: trimesh.Trimesh, resolution: int = 220) -> trimesh.Trimesh:
    """Last-resort repair: rebuild the surface from a filled voxel grid (always watertight).

    Loses fine detail below ``extent / resolution``; only used when hole filling cannot close a mesh.
    """
    pitch = float(np.max(mesh.extents)) / resolution
    grid = mesh.voxelized(pitch).fill()
    rebuilt = grid.marching_cubes
    rebuilt.apply_transform(grid.transform)
    return rebuilt


def make_watertight(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Return a closed, consistently oriented mesh plus a human-readable log of what was done."""
    log: list[str] = []
    out = clean(mesh)
    out, filled = fill_holes(out)
    if filled:
        log.append(f"Filled {filled} hole{'s' if filled != 1 else ''}")
    out = fix_orientation(clean(out))
    if not out.is_watertight:
        log.append("Hole filling was not enough; rebuilt the surface from a voxel grid")
        out = fix_orientation(voxel_remesh(out))
    return out, log
