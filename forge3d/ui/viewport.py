"""3D viewport: orbit/pan/zoom view of the model on a print bed, with a support-overhang overlay."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
import trimesh
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader
from PySide6 import QtCore, QtGui, QtWidgets

from ..meshops import overhang_mask

MAX_DISPLAY_FACES = 220_000  # the GPU copes with more, but the CPU-side unwelding for colours would not

MODEL_COLOR = (0.62, 0.70, 0.80, 1.0)
OVERHANG_COLOR = (0.90, 0.25, 0.22, 1.0)

# Lit by a key light, a fill light and a sky/ground ambient term. Back faces are tinted magenta, which makes
# inside-out or open surfaces (things a slicer would choke on) obvious at a glance.
_SHADER_NAME = "forge3d_lit"
ShaderProgram(
    _SHADER_NAME,
    [
        VertexShader(
            """
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
            """
        ),
        FragmentShader(
            """
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                vec3 n = normalize(v_normal);
                bool back = !gl_FrontFacing;
                if (back) n = -n;
                float key  = max(dot(n, normalize(vec3(0.45, 0.65, 0.60))), 0.0);
                float fill = max(dot(n, normalize(vec3(-0.60, -0.20, 0.50))), 0.0);
                float sky  = 0.5 + 0.5 * n.y;
                vec3 rgb = v_color.rgb * (0.28 + 0.58 * key + 0.22 * fill + 0.18 * sky);
                if (back) rgb = mix(rgb, vec3(0.95, 0.15, 0.85), 0.55);
                gl_FragColor = vec4(rgb, v_color.a);
            }
            """
        ),
    ],
)


def _forget_compiled_shaders() -> None:
    """Drop every shader program pyqtgraph has cached (they are only valid in the context that built them)."""
    for program in ShaderProgram.names.values():
        program.prog = None
    for cls in (gl.GLLinePlotItem, gl.GLScatterPlotItem, gl.GLImageItem, gl.GLVolumeItem):
        if hasattr(cls, "_shaderProgram"):
            cls._shaderProgram = None  # class-level caches used by the non-mesh items


def _simplified_for_display(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) <= MAX_DISPLAY_FACES:
        return mesh
    import fast_simplification

    points, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=1.0 - MAX_DISPLAY_FACES / len(mesh.faces),
    )
    return trimesh.Trimesh(points, faces, process=False)


class Viewport(gl.GLViewWidget):
    """Z-up view. Left-drag orbits, right/middle-drag pans, the wheel zooms."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((27, 30, 36))
        self.opts["fov"] = 35
        self._mesh_item: gl.GLMeshItem | None = None
        self._bed_items: list = []
        self._bed = (220.0, 220.0)
        self._bounds: np.ndarray | None = None
        self._last_view: dict | None = None  # what to redraw if the GL context is re-created
        self._gl_initialised = False
        self.set_bed(*self._bed)
        self.reset_camera()

        self._hint = QtWidgets.QLabel(self)
        self._hint.setAlignment(QtCore.Qt.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #6b7480; font-size: 15px; background: transparent;")
        self._hint.setText(
            "Describe an object and press <b>Draw pictures</b>,<br>open your own picture,<br>"
            "or drop an STL / OBJ / GLB file here."
        )
        self._hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    # ------------------------------------------------------------------ GL context lifetime

    def initializeGL(self) -> None:
        """Called by Qt for every new OpenGL context, not just the first.

        Compiled shader programs and GPU buffers belong to the context that made them. pyqtgraph caches
        programs globally, so after Qt re-creates the context (the window moved to another monitor or GPU,
        or a second window opened) the cached ids are invalid and drawing fails. Forget them and rebuild.
        """
        super().initializeGL()
        _forget_compiled_shaders()
        if self._gl_initialised:
            QtCore.QTimer.singleShot(0, self._rebuild_scene)
        self._gl_initialised = True

    def _rebuild_scene(self) -> None:
        self.set_bed(*self._bed)
        if self._last_view is not None:
            self.show_mesh(**self._last_view)

    # ------------------------------------------------------------------ layout

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._hint.setGeometry(self.rect())

    # ------------------------------------------------------------------ bed

    def set_bed(self, size_x: float, size_y: float) -> None:
        for item in self._bed_items:
            self.removeItem(item)
        self._bed_items = []
        self._bed = (float(size_x), float(size_y))

        grid = gl.GLGridItem()
        grid.setSize(size_x, size_y, 0)
        grid.setSpacing(10, 10, 10)
        grid.setColor((255, 255, 255, 34))
        coarse = gl.GLGridItem()
        coarse.setSize(size_x, size_y, 0)
        coarse.setSpacing(50, 50, 50)
        coarse.setColor((255, 255, 255, 70))
        hx, hy = size_x / 2, size_y / 2
        outline = gl.GLLinePlotItem(
            pos=np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0]], dtype=np.float32),
            color=(0.30, 0.61, 0.91, 1.0),
            width=2,
            antialias=True,
        )
        for item in (grid, coarse, outline):
            self.addItem(item)
            self._bed_items.append(item)

    # ------------------------------------------------------------------ mesh

    def clear_mesh(self) -> None:
        if self._mesh_item is not None:
            self.removeItem(self._mesh_item)
            self._mesh_item = None
        self._bounds = None
        self._hint.show()

    def show_mesh(
        self,
        mesh: trimesh.Trimesh | None,
        *,
        overhang_deg: float | None = None,
        wireframe: bool = False,
        frame: bool = False,
    ) -> None:
        """Display ``mesh`` (print frame, mm). ``overhang_deg`` turns on the support overlay."""
        if mesh is None or len(mesh.faces) == 0:
            self._last_view = None
            self.clear_mesh()
            return
        self._last_view = {"mesh": mesh, "overhang_deg": overhang_deg, "wireframe": wireframe, "frame": False}
        if self._mesh_item is not None:
            self.removeItem(self._mesh_item)
            self._mesh_item = None

        display = _simplified_for_display(mesh)
        if overhang_deg is not None:
            mask = overhang_mask(display, overhang_deg)
            colors = np.tile(np.array(MODEL_COLOR, dtype=np.float32), (len(display.faces), 1))
            colors[mask] = OVERHANG_COLOR
            # per-face colours need un-shared vertices
            data = gl.MeshData(
                vertexes=np.asarray(display.vertices[display.faces], dtype=np.float32).reshape(-1, 3),
                faces=np.arange(len(display.faces) * 3, dtype=np.uint32).reshape(-1, 3),
                faceColors=colors,
            )
            item = gl.GLMeshItem(meshdata=data, smooth=False, shader=_SHADER_NAME, drawEdges=wireframe)
        else:
            data = gl.MeshData(
                vertexes=np.asarray(display.vertices, dtype=np.float32), faces=np.asarray(display.faces, dtype=np.uint32)
            )
            item = gl.GLMeshItem(
                meshdata=data, smooth=True, color=MODEL_COLOR, shader=_SHADER_NAME, drawEdges=wireframe
            )
        if wireframe:
            item.opts["edgeColor"] = (0.05, 0.07, 0.09, 0.6)
        self.addItem(item)
        self._mesh_item = item
        self._hint.hide()

        first = self._bounds is None
        self._bounds = np.asarray(mesh.bounds)
        if frame or first:
            self.reset_camera()
        self.update()

    # ------------------------------------------------------------------ camera

    def reset_camera(self) -> None:
        if self._bounds is not None:
            centre = self._bounds.mean(axis=0)
            radius = float(np.linalg.norm(self._bounds[1] - self._bounds[0])) / 2
        else:
            centre, radius = np.array([0.0, 0.0, 30.0]), 90.0
        self.opts["center"] = QtGui.QVector3D(*[float(c) for c in centre])
        self.opts["distance"] = max(radius * 4.2, 40.0)
        self.opts["elevation"] = 24
        self.opts["azimuth"] = -55
        self.update()

    def snapshot(self) -> QtGui.QImage:
        return self.grabFramebuffer()
