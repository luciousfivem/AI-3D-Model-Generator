import numpy as np
import pytest
import trimesh

from forge3d import meshops


def sphere(radius=20.0, subdivisions=4):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


# --------------------------------------------------------------------------- repair


def test_clean_drops_degenerate_and_duplicate_faces():
    mesh = sphere(subdivisions=2)
    faces = np.vstack([mesh.faces, mesh.faces[:5], [[0, 0, 1]]])  # duplicates + one degenerate
    dirty = trimesh.Trimesh(mesh.vertices, faces, process=False)
    cleaned = meshops.clean(dirty)
    assert len(cleaned.faces) == len(mesh.faces)
    assert cleaned.is_watertight


def test_remove_floaters_drops_tiny_pieces_only():
    body = sphere(20.0)
    crumb = sphere(0.5, subdivisions=1)
    crumb.apply_translation([40, 0, 0])
    combined = trimesh.util.concatenate([body, crumb])
    result, removed = meshops.remove_floaters(combined)
    assert removed == 1
    assert len(result.faces) == len(body.faces)


def test_remove_floaters_keeps_two_comparable_pieces():
    a, b = sphere(10.0), sphere(10.0)
    b.apply_translation([40, 0, 0])
    _, removed = meshops.remove_floaters(trimesh.util.concatenate([a, b]))
    assert removed == 0


def test_fill_holes_closes_a_punctured_sphere():
    mesh = sphere(subdivisions=3)
    keep = np.ones(len(mesh.faces), bool)
    keep[np.argsort(mesh.triangles_center[:, 2])[-12:]] = False  # punch out the top cap
    holed = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    holed.remove_unreferenced_vertices()
    assert not holed.is_watertight

    filled, count = meshops.fill_holes(holed)
    assert count >= 1
    filled = meshops.fix_orientation(meshops.clean(filled))
    assert filled.is_watertight
    assert filled.volume == pytest.approx(mesh.volume, rel=0.05)


def test_fill_holes_closes_a_large_cut_open_base():
    mesh = sphere(subdivisions=4)
    cut = mesh.slice_plane([0, 0, -10], [0, 0, 1], cap=False)  # open bottom, many boundary vertices
    assert not cut.is_watertight
    filled, count = meshops.fill_holes(cut)
    filled = meshops.fix_orientation(meshops.clean(filled))
    assert count == 1
    assert filled.is_watertight and filled.volume > 0


def test_make_watertight_fixes_inside_out_mesh():
    mesh = sphere()
    mesh.invert()
    fixed, _ = meshops.make_watertight(mesh)
    assert fixed.is_watertight and fixed.volume > 0


def test_voxel_remesh_is_always_closed():
    mesh = sphere(subdivisions=3)
    rebuilt = meshops.voxel_remesh(mesh, resolution=64)
    assert rebuilt.is_watertight
    assert rebuilt.volume == pytest.approx(mesh.volume, rel=0.1)


# --------------------------------------------------------------------------- orient


@pytest.mark.parametrize("up", meshops.UP_AXES)
def test_up_axis_rotation_maps_axis_to_z(up):
    vector = np.zeros(3)
    vector["xyz".index(up[-1])] = -1.0 if up.startswith("-") else 1.0
    assert meshops.up_axis_rotation(up) @ vector == pytest.approx([0, 0, 1])
    assert np.linalg.det(meshops.up_axis_rotation(up)) == pytest.approx(1.0)  # a rotation, not a mirror


def test_scale_and_drop_to_bed():
    mesh = sphere(20.0)
    mesh.apply_translation([100, -50, 7])
    placed = meshops.drop_to_bed(meshops.scale_to_height(mesh, 30.0))
    assert placed.extents[2] == pytest.approx(30.0, rel=1e-6)
    assert placed.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
    assert placed.bounds.mean(axis=0)[:2] == pytest.approx([0, 0], abs=1e-6)


def test_a_steep_inverted_cone_needs_no_support():
    cone = trimesh.creation.cone(radius=10, height=30)  # walls only ~18 degrees from vertical
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # apex down
    assert meshops.overhang_mask(cone).sum() == 0


def test_auto_orient_stands_a_cone_on_its_base():
    cone = trimesh.creation.cone(radius=30, height=10)  # wide and flat: apex-down walls are ~72 degrees
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # apex down
    before = meshops.overhang_mask(cone).mean()

    rotation, info = meshops.auto_orient(cone)
    oriented = meshops.rotate(cone, rotation)
    after_mask = meshops.overhang_mask(oriented)
    assert after_mask.sum() == 0, "cone on its base needs no supports"
    assert before > 0.3
    assert info["contact_fraction"] > 0


def test_overhang_mask_ignores_faces_on_the_bed():
    box = trimesh.creation.box([20, 20, 20])
    box.apply_translation([0, 0, 10])
    assert meshops.overhang_mask(box).sum() == 0  # bottom face sits on the bed


# --------------------------------------------------------------------------- analyze


def test_analyze_a_clean_sphere():
    mesh = meshops.drop_to_bed(sphere(20.0))
    report = meshops.analyze(mesh)
    assert report.watertight and report.printable and report.fits_bed
    assert report.volume_mm3 == pytest.approx(4 / 3 * np.pi * 20**3, rel=0.01)
    assert report.weight_g == pytest.approx(report.volume_mm3 / 1000 * 1.24, rel=1e-6)
    assert report.n_components == 1


def test_analyze_flags_open_mesh_and_oversize():
    mesh = meshops.drop_to_bed(sphere(200.0, subdivisions=3))
    mesh.update_faces(np.arange(len(mesh.faces)) % 10 != 0)  # punch holes
    report = meshops.analyze(mesh, bed_mm=(220, 220, 250))
    codes = {i.code for i in report.issues}
    assert {"not_watertight", "too_big"} <= codes
    assert not report.printable


def test_analyze_reports_loose_pieces():
    a, b = sphere(10.0), sphere(10.0)
    b.apply_translation([40, 0, 0])
    report = meshops.analyze(meshops.drop_to_bed(trimesh.util.concatenate([a, b])))
    assert report.n_components == 2
    assert any(i.code == "components" for i in report.issues)


def test_wall_thickness_solid_sphere_is_its_diameter():
    distances = meshops.measure_wall_thickness(sphere(20.0), samples=300)
    assert len(distances) > 200
    assert np.median(distances) == pytest.approx(40.0, rel=0.1)


def test_wall_thickness_flags_thin_shell():
    outer = sphere(20.0, subdivisions=3)
    inner = sphere(19.7, subdivisions=3)
    inner.invert()
    shell = trimesh.util.concatenate([outer, inner])
    report = meshops.analyze(
        meshops.drop_to_bed(shell), measure_thickness=True, min_wall_mm=0.8
    )
    assert report.thin_pct is not None and report.thin_pct > 50
    assert any(i.code == "thin_walls" for i in report.issues)


# --------------------------------------------------------------------------- shape


def test_decimate_hits_roughly_the_target():
    mesh = sphere(subdivisions=5)  # 20480 faces
    small = meshops.decimate(mesh, 2000)
    assert 1500 < len(small.faces) < 2600
    assert small.volume == pytest.approx(mesh.volume, rel=0.05)


def test_smooth_preserves_volume():
    mesh = sphere(subdivisions=4)
    rough = trimesh.Trimesh(mesh.vertices * (1 + 0.02 * np.random.default_rng(1).normal(size=(len(mesh.vertices), 1))), mesh.faces)
    smoothed = meshops.smooth(rough, 5)
    assert smoothed.volume == pytest.approx(rough.volume, rel=0.02)  # no Laplacian shrinkage
    assert smoothed.area < rough.area  # but it really did smooth the noise out
    assert meshops.smooth(mesh, 0) is mesh


def test_cut_flat_base_makes_a_closed_flat_bottom():
    mesh = meshops.drop_to_bed(sphere(20.0))
    cut = meshops.cut_flat_base(mesh, 3.0)
    cut = meshops.drop_to_bed(cut)
    assert cut.is_watertight
    assert cut.extents[2] == pytest.approx(mesh.extents[2] - 3.0, rel=0.01)
    flat = cut.area_faces[cut.triangles_center[:, 2] < 1e-6].sum()
    assert flat > 100  # a real contact patch (pi * ~11^2 ~ 380 mm^2)


def test_cut_flat_base_refuses_to_eat_the_model():
    mesh = meshops.drop_to_bed(sphere(20.0))
    assert meshops.cut_flat_base(mesh, 30.0) is mesh


# --------------------------------------------------------------------------- export / prepare


@pytest.mark.parametrize("ext", list(meshops.FORMATS))
def test_export_round_trip(tmp_path, ext):
    mesh = meshops.drop_to_bed(sphere(15.0, subdivisions=3))
    path = meshops.export_mesh(mesh, tmp_path / f"model.{ext}")
    assert path.exists() and path.stat().st_size > 0
    again = meshops.load_mesh(path)
    assert again.is_watertight
    assert again.volume == pytest.approx(mesh.volume, rel=0.01)


def test_export_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        meshops.export_mesh(sphere(), tmp_path / "x.xyz")


def test_stl_is_binary_and_millimetre_sized(tmp_path):
    mesh = meshops.drop_to_bed(meshops.scale_to_height(sphere(), 42.0))
    path = meshops.export_mesh(mesh, tmp_path / "m.stl")
    assert not path.read_bytes()[:5] == b"solid"  # binary STL does not start with "solid"
    assert meshops.load_mesh(path).extents[2] == pytest.approx(42.0, rel=1e-4)


def test_prepare_turns_a_yup_model_into_a_bed_ready_one():
    capsule = trimesh.creation.capsule(height=40, radius=10)  # long axis is Z
    capsule.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # now long axis is Y
    options = meshops.PrepOptions(source_up="y", height_mm=75.0)
    out = meshops.prepare(capsule, options)
    assert out.extents[2] == pytest.approx(75.0, rel=1e-6)  # tall along Z, not along Y
    assert out.extents[2] > out.extents[0]
    assert out.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
    assert out.is_watertight


def test_prepare_logs_what_it_did():
    body = sphere(20.0)
    crumb = sphere(0.4, subdivisions=1)
    crumb.apply_translation([50, 0, 0])
    messages = []
    out = meshops.prepare(
        trimesh.util.concatenate([body, crumb]),
        meshops.PrepOptions(source_up="z", height_mm=None, decimate_faces=1000),
        messages.append,
    )
    assert any("loose fragment" in m for m in messages)
    assert any("Reduced" in m for m in messages)
    assert len(meshops.analyze(out).issues) >= 1
    assert out.is_watertight


def test_repair_stage_key_ignores_orientation_changes():
    a = meshops.PrepOptions()
    b = meshops.PrepOptions(rotation=meshops.axis_rotation("z", 90), height_mm=10)
    assert a.repair_key() == b.repair_key()
    assert a.repair_key() != meshops.PrepOptions(smooth_iterations=3).repair_key()
