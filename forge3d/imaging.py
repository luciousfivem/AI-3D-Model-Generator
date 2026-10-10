"""Input-image preparation: cut the subject out and centre it, as image-to-3D models expect."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_REMBG_SESSION = None
MAX_SIDE = 1024


def has_cutout(image: Image.Image) -> bool:
    """True if the image already carries a real transparent background."""
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel("A"))
    return float((alpha < 250).mean()) > 0.01


def _fallback_cutout(image: Image.Image, tolerance: float = 34.0) -> Image.Image:
    """Dependency-free background removal for plain backgrounds (e.g. generated white-backdrop renders).

    Treats pixels close to the median border colour as background, keeping only background regions that
    actually touch the image border so the subject's own bright areas survive.
    """
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    close = np.linalg.norm(rgb - background, axis=2) < tolerance
    labels, _ = ndimage.label(close)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    is_background = np.isin(labels, edge_labels[edge_labels > 0])
    alpha = Image.fromarray(np.where(is_background, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    out = image.convert("RGBA")
    out.putalpha(alpha)
    return out


def remove_background(image: Image.Image) -> Image.Image:
    """Return an RGBA cut-out. Uses rembg when installed (best), else a simple border-colour method."""
    global _REMBG_SESSION
    if has_cutout(image):
        return image
    try:
        from rembg import new_session, remove

        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session("u2net")
        return remove(image.convert("RGB"), session=_REMBG_SESSION).convert("RGBA")
    except Exception:
        # rembg missing, offline first run, or an onnxruntime problem: degrade gracefully
        return _fallback_cutout(image)


def crop_to_subject(rgba: Image.Image, margin: float = 0.08) -> Image.Image:
    """Crop to the opaque bounding box and pad to a square with a little breathing room."""
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        raise ValueError("No subject found in the image (it looks empty or fully transparent).")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    subject = rgba.crop(box)
    side = int(max(subject.size) * (1 + 2 * margin))
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.paste(subject, ((side - subject.width) // 2, (side - subject.height) // 2))
    if side > MAX_SIDE:
        canvas = canvas.resize((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return canvas


def prepare_input(image: Image.Image, remove_bg: bool = True) -> Image.Image:
    """Image as handed to the 3D model: RGBA, subject centred, background transparent."""
    image = image.convert("RGBA") if image.mode not in ("RGB", "RGBA") else image
    if remove_bg or has_cutout(image):
        image = remove_background(image)
    else:
        image = image.convert("RGBA")
    return crop_to_subject(image)
