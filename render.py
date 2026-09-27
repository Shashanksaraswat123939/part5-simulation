"""
render.py -- pictures of an assembled car from a run_car output folder.

    python render.py results/            # -> results/car.png

Four views (side, top, front, iso) of body + every Part 4 patch, both halves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

COLOURS = {"body": "#9aa7b4", "wheelF": "#222222", "wheelR": "#222222", "supports": "#d0862f",
           "halo": "#c03a2b", "fwing": "#2b6cb0", "rwing": "#2b6cb0", "tethers": "#6b8e23",
           "nose": "#7f5aa2"}


def _load(out: Path):
    import trimesh
    body = out / ("body_final_half.stl" if (out / "body_final_half.stl").exists() else "body_half.stl")
    parts = out / ("parts_final" if (out / "parts_final").exists() else "parts")
    asm = json.loads((parts / "assembly.json").read_text())
    meshes = {"body": trimesh.load(str(body), force="mesh")}
    for s in asm["extra_surfaces"]:
        p = Path(s["stl"])
        p = p if p.exists() else parts / p.name
        meshes[s["name"]] = trimesh.load(str(p), force="mesh")
    full = {}
    for k, m in meshes.items():
        mir = m.copy()
        mir.vertices[:, 1] *= -1
        mir.invert()
        f = trimesh.util.concatenate([m, mir])
        if len(f.faces) > 6000:
            try:
                f = f.simplify_quadric_decimation(face_count=6000)
            except Exception:  # noqa: BLE001 -- a picture must not fail on a decimator
                pass
        full[k] = f
    return full


def render(out: Path, png: Path = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    full = _load(out)
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    views = [("side", 0, -90), ("top", 90, -90), ("front", 0, 180), ("iso", 25, -125)]
    fig = plt.figure(figsize=(16, 9))
    allv = np.vstack([m.vertices for m in full.values()]) * 1e3
    lo, hi = allv.min(0), allv.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    for n, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, n + 1, projection="3d", proj_type="ortho")
        for k, m in full.items():
            tri = m.vertices[m.faces] * 1e3
            shade = 0.45 + 0.55 * np.clip(m.face_normals @ light, 0, 1)
            base = np.array(matplotlib.colors.to_rgb(COLOURS.get(k, "#888888")))
            ax.add_collection3d(Poly3DCollection(tri, facecolors=base * shade[:, None],
                                                 edgecolors="none"))
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(el, az)
        ax.set_axis_off()
        ax.set_title(name)
    png = png or out / "car.png"
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
