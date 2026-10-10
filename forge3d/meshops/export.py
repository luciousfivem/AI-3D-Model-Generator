"""Writing meshes for slicers and other tools."""

from __future__ import annotations

from pathlib import Path

import trimesh

# extension -> (label, note)
FORMATS = {
    "stl": ("STL (binary)", "Universal slicer format. Units are assumed to be millimetres."),
    "3mf": ("3MF", "Modern slicer format that stores millimetre units. Preferred by PrusaSlicer, Bambu, Cura."),
    "obj": ("OBJ", "For Blender and other modelling tools."),
    "ply": ("PLY", "Simple interchange format."),
    "glb": ("GLB", "For game engines and web viewers (glTF binary)."),
}


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write ``mesh`` to ``path``; the format comes from the file extension."""
    path = Path(path)
    extension = path.suffix.lower().lstrip(".")
    if extension not in FORMATS:
        raise ValueError(f"Unsupported export format '.{extension}'. Choose one of: {', '.join(FORMATS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type=extension)
    return path


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load any supported mesh file as a single Trimesh (scenes are merged)."""
    # process=True merges duplicate vertices; STL stores every triangle's corners separately, so without
    # it a perfectly closed STL would look full of holes.
    loaded = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"'{Path(path).name}' does not contain a triangle mesh")
    return loaded
