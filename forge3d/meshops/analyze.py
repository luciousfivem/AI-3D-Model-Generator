"""Printability analysis: is this mesh going to slice and print, and what will it cost?"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .orient import overhang_mask


@dataclass
class Issue:
    severity: str  # "error" | "warn" | "info"
    code: str
    message: str


@dataclass
class PrintReport:
    n_vertices: int = 0
    n_faces: int = 0
    n_components: int = 0
    watertight: bool = False
    winding_consistent: bool = False
    size_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_mm3: float = 0.0
    area_mm2: float = 0.0
    weight_g: float = 0.0  # 100 % solid; real prints with infill weigh less
    overhang_pct: float = 0.0  # share of surface area needing support
    fits_bed: bool = True
    thin_pct: float | None = None  # share of sampled surface thinner than min_wall (None = not measured)
    median_wall_mm: float | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def printable(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def measure_wall_thickness(
    mesh: trimesh.Trimesh, samples: int = 1500, max_faces: int = 40000, seed: int = 0
) -> np.ndarray:
    """Estimate local wall thickness by shooting rays inward from random surface points.

    Works on a decimated copy so it stays interactive. Returns one distance (mm) per ray that hit the
    opposite wall; rays that escape (open mesh) are dropped.
    """
    probe = mesh
    if len(mesh.faces) > max_faces:
        import fast_simplification

        points, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            target_reduction=1.0 - max_faces / len(mesh.faces),
        )
        probe = trimesh.Trimesh(points, faces, process=False)

    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(probe, samples, seed=rng)
    normals = probe.face_normals[face_index]
    nudge = max(float(np.max(probe.extents)) * 1e-4, 1e-6)
    origins = points - normals * nudge
    directions = -normals

    locations, ray_index, _ = probe.ray.intersects_location(origins, directions, multiple_hits=False)
    if len(ray_index) == 0:
        return np.empty(0)
    return np.linalg.norm(locations - origins[ray_index], axis=1)


def analyze(
    mesh: trimesh.Trimesh,
    *,
    bed_mm: tuple[float, float, float] = (220.0, 220.0, 250.0),
    max_overhang_deg: float = 45.0,
    min_wall_mm: float = 0.8,
    density_g_cm3: float = 1.24,
    measure_thickness: bool = False,
) -> PrintReport:
    """Inspect a mesh that is already in the print frame (+Z up, mm)."""
    report = PrintReport(
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        n_components=len(mesh.split(only_watertight=False)) if len(mesh.faces) else 0,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        size_mm=tuple(float(v) for v in mesh.extents),
        area_mm2=float(mesh.area),
    )
    issues = report.issues

    if report.watertight and mesh.is_volume:
        report.volume_mm3 = float(mesh.volume)
        report.weight_g = report.volume_mm3 / 1000.0 * density_g_cm3
    else:
        issues.append(
            Issue("error", "not_watertight", "Mesh has holes or non-manifold edges; slicers may fail or print gaps.")
        )
    if report.watertight and not report.winding_consistent:
        issues.append(Issue("error", "winding", "Face directions are inconsistent (inside-out faces)."))

    if report.n_components > 1:
        issues.append(
            Issue("warn", "components", f"{report.n_components} separate pieces; loose pieces will print detached.")
        )

    sx, sy, sz = report.size_mm
    report.fits_bed = sx <= bed_mm[0] and sy <= bed_mm[1] and sz <= bed_mm[2]
    if not report.fits_bed:
        issues.append(
            Issue(
                "error",
                "too_big",
                f"{sx:.0f} x {sy:.0f} x {sz:.0f} mm does not fit the {bed_mm[0]:.0f} x {bed_mm[1]:.0f} x "
                f"{bed_mm[2]:.0f} mm bed. Scale it down or split it.",
            )
        )

    if len(mesh.faces):
        mask = overhang_mask(mesh, max_overhang_deg)
        report.overhang_pct = 100.0 * float(mesh.area_faces[mask].sum()) / (report.area_mm2 or 1.0)
        if report.overhang_pct > 15.0:
            issues.append(
                Issue(
                    "warn",
                    "overhangs",
                    f"{report.overhang_pct:.0f}% of the surface overhangs more than {max_overhang_deg:.0f} degrees; "
                    "expect supports (try Auto-orient).",
                )
            )

    if measure_thickness and len(mesh.faces):
        try:
            distances = measure_wall_thickness(mesh)
        except Exception as error:  # ray backends vary by platform; thickness is advisory only
            issues.append(Issue("info", "thickness_unavailable", f"Wall-thickness check skipped: {error}"))
            distances = np.empty(0)
        if len(distances):
            report.thin_pct = 100.0 * float((distances < min_wall_mm).mean())
            report.median_wall_mm = float(np.median(distances))
            if report.thin_pct > 5.0:
                issues.append(
                    Issue(
                        "warn",
                        "thin_walls",
                        f"About {report.thin_pct:.0f}% of the surface is thinner than {min_wall_mm:.1f} mm and may "
                        "not print. Scale the model up or thicken fine details.",
                    )
                )

    if not issues:
        issues.append(Issue("info", "ok", "No problems found."))
    return report
