"""Mesh processing for 3D printing (pure numpy/trimesh, no AI or GUI dependencies)."""

from .analyze import Issue, PrintReport, analyze, measure_wall_thickness
from .export import FORMATS, export_mesh, load_mesh
from .orient import (
    UP_AXES,
    auto_orient,
    axis_rotation,
    drop_to_bed,
    overhang_mask,
    rotate,
    scale_to_height,
    scale_to_largest_dimension,
    up_axis_rotation,
)
from .prepare import PrepOptions, choose_orientation, placement_stage, prepare, repair_stage
from .repair import clean, fill_holes, fix_orientation, make_watertight, remove_floaters, voxel_remesh
from .shape import cut_flat_base, decimate, smooth

__all__ = [
    "FORMATS",
    "Issue",
    "PrepOptions",
    "PrintReport",
    "UP_AXES",
    "analyze",
    "auto_orient",
    "axis_rotation",
    "choose_orientation",
    "clean",
    "cut_flat_base",
    "decimate",
    "drop_to_bed",
    "export_mesh",
    "fill_holes",
    "fix_orientation",
    "load_mesh",
    "make_watertight",
    "measure_wall_thickness",
    "overhang_mask",
    "placement_stage",
    "prepare",
    "remove_floaters",
    "repair_stage",
    "rotate",
    "scale_to_height",
    "scale_to_largest_dimension",
    "smooth",
    "up_axis_rotation",
    "voxel_remesh",
]
