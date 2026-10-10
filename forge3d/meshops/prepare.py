"""The print-prep pipeline: raw generated mesh -> bed-ready, slicer-clean mesh.

Split into two stages so interactive edits stay fast:

* stage A (:func:`repair_stage`) is orientation independent and relatively slow
  (clean-up, floaters, watertight, smoothing, decimation). Cache its result.
* stage B (:func:`placement_stage`) is cheap: rotate, scale to a real size, flat base, drop to bed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import trimesh

from . import orient, repair, shape

Log = Callable[[str], None]


@dataclass
class PrepOptions:
    source_up: str = "y"  # up axis of the generated mesh (glTF-style models are Y-up)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # user rotation, in the print frame
    height_mm: float | None = 60.0  # None keeps the mesh's own scale
    remove_floaters: bool = True
    make_watertight: bool = True
    smooth_iterations: int = 0
    decimate_faces: int | None = None
    flat_base_mm: float = 0.0

    def repair_key(self) -> tuple:
        """Everything stage A depends on (used as a cache key)."""
        return (self.remove_floaters, self.make_watertight, self.smooth_iterations, self.decimate_faces)


def repair_stage(raw: trimesh.Trimesh, options: PrepOptions, log: Log | None = None) -> trimesh.Trimesh:
    note = log or (lambda message: None)
    mesh = repair.clean(raw)

    if options.remove_floaters:
        mesh, removed = repair.remove_floaters(mesh)
        if removed:
            note(f"Removed {removed} loose fragment{'s' if removed != 1 else ''}")

    if options.make_watertight:
        mesh, actions = repair.make_watertight(mesh)
        for action in actions:
            note(action)

    if options.smooth_iterations > 0:
        mesh = shape.smooth(mesh, options.smooth_iterations)
        note(f"Smoothed ({options.smooth_iterations} passes)")

    if options.decimate_faces and options.decimate_faces < len(mesh.faces):
        before = len(mesh.faces)
        mesh = shape.decimate(mesh, options.decimate_faces)
        note(f"Reduced {before:,} -> {len(mesh.faces):,} triangles")
        if options.make_watertight and not mesh.is_watertight:
            mesh, actions = repair.make_watertight(mesh)
            for action in actions:
                note(action)
    return mesh


def choose_orientation(
    repaired: trimesh.Trimesh, options: PrepOptions, max_overhang_deg: float = 45.0
) -> tuple[np.ndarray, dict]:
    """Best print orientation for a stage-A mesh: returns ``(rotation, info)`` for ``options.rotation``.

    The search happens in the print frame (after the source up-axis is applied), so the returned
    rotation composes correctly with :func:`placement_stage`.
    """
    probe = repaired.copy()
    transform = np.eye(4)
    transform[:3, :3] = orient.up_axis_rotation(options.source_up)
    probe.apply_transform(transform)
    return orient.auto_orient(probe, max_overhang_deg)


def placement_stage(mesh: trimesh.Trimesh, options: PrepOptions) -> trimesh.Trimesh:
    rotation = options.rotation @ orient.up_axis_rotation(options.source_up)
    out = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    out.apply_transform(transform)
    out = orient.drop_to_bed(out)

    if options.height_mm:
        out = orient.scale_to_height(out, options.height_mm)
        out = orient.drop_to_bed(out)

    if options.flat_base_mm > 0:
        out = orient.drop_to_bed(shape.cut_flat_base(out, options.flat_base_mm))
    return out


def prepare(raw: trimesh.Trimesh, options: PrepOptions, log: Log | None = None) -> trimesh.Trimesh:
    return placement_stage(repair_stage(raw, options, log), options)
