# Third-party models and libraries

Forge3D's own code is MIT-licensed. It *uses* the following. **Read the model licenses before using generated
output for anything other than personal projects.** The two AI-model licenses were checked against the model
repositories when this was written (October 2026). The library licenses are listed from general knowledge and were
not re-verified. The authoritative text is always in each project's own repository.

## AI models (downloaded on demand, not distributed with this project)

| Model | Used for | License |
|---|---|---|
| [Hunyuan3D-2 / 2mini](https://huggingface.co/tencent/Hunyuan3D-2) (Tencent) | image → 3D shape | **Tencent Hunyuan Non-Commercial License.** Not licensed in the EU, UK or South Korea. The `hy3dgen` code is under the same license. |
| [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) (Stability AI) | text → picture | **Stability AI Non-Commercial Research Community License** (`sai-nc-community`). |
| [U²-Net](https://github.com/xuebinqin/U-2-Net) via [rembg](https://github.com/danielgatis/rembg) | background removal | Apache-2.0 (model), MIT (rembg) |

## Libraries

PySide6 (LGPL-3.0), pyqtgraph (MIT), PyOpenGL (BSD), trimesh (MIT), numpy / scipy / scikit-image (BSD),
fast-simplification (MIT), networkx (BSD), mapbox-earcut (ISC), Pillow (HPND), platformdirs (MIT),
PyTorch (BSD), diffusers / transformers / accelerate / huggingface_hub (Apache-2.0), einops (MIT),
omegaconf (BSD), onnxruntime (MIT), OpenCV (Apache-2.0).

`hy3dgen` imports **PyMeshLab (GPL-3.0)**. Keep this in mind if you redistribute a bundled build of the app.
