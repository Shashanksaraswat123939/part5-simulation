"""
legality.py -- the assembled car, measured the way a scrutineer measures it.

Part 1's hard masks keep the body legal DURING descent; this module is the
independent audit on the finished geometry. Two of those masks were found wrong
on 2026-09-25 (block envelope, T5.5 wall), which is exactly why the audit exists.

    check(body_half_stl, assembly_json, mass_state) -> {rule: {margin, pass, note}}

Margins are in mm (or g for mass). margin >= 0 passes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import paths  # noqa: F401

REGS = {
    "T3.4_width_min": 65.0, "T3.4_width_max": 85.0, "T3.5_height": 65.0,
    "T3.7_clearance": 1.5, "T3.6_mass_g": 48.0, "T7.3_W_min": 120.0, "T7.3_W_max": 140.0,
    "T8.2_nose_overhang": 40.0, "T8.5.1_nose_z": 25.0, "T8.5.1_nose_half_width": 15.0,
    "T9.4.2_rear_overhang": 40.0, "T5.5_wall": 3.0,
    "block_length": 223.0, "block_width": 65.0, "block_height": 50.0,
}


def _entry(margin: float, note: str = "") -> dict:
    return {"margin": float(margin), "pass": bool(margin >= -1e-6), "note": note}


def _mirror(mesh):
    m = mesh.copy()
    m.vertices[:, 1] *= -1
    return m


def t55_wall_fraction(body_half, rear_face_mm: float, depth_mm: float = 50.0,
                      bore_r_mm: float = 9.0, axis_z_mm: float = 35.0) -> float:
    """Fraction of probe points inside the body in the 3 mm annulus around the
    chamber, over the 45 mm minimum depth (T5.5). 1.0 = a full 3 mm wall."""
    xs = np.linspace(rear_face_mm - depth_mm + 2.0, rear_face_mm - depth_mm + 45.0, 12)
    rs = (bore_r_mm + 0.5, bore_r_mm + 1.5, bore_r_mm + 2.5)
    # Right half, but NOT on the symmetry plane itself: a containment test on
    # the capped y=0 face is ambiguous and reported false misses at 0/180 deg.
    ang = np.linspace(0.08, math.pi - 0.08, 13)
    pts = np.array([(x, r * math.sin(a), axis_z_mm + r * math.cos(a))
                    for x in xs for r in rs for a in ang]) / 1e3
    pts[:, 1] = np.maximum(pts[:, 1], 1e-6)
    inside = body_half.contains(pts)
    return float(inside.mean())


def check(body_half_stl: str, assembly: dict, mass_state: dict | None = None,
          field_bodies: int | None = None) -> dict:
    import trimesh
    W, xf = assembly["W_mm"], assembly["x_front_mm"]
    ref_a, ref_b = xf - 16.0, xf + W + 16.0
    body = trimesh.load(str(body_half_stl), force="mesh")
    parts = {s["name"]: trimesh.load(s["stl"], force="mesh") for s in assembly["extra_surfaces"]}
    everything = [body] + list(parts.values())
    non_wheel = [body] + [m for n, m in parts.items() if not n.startswith("wheel")]
    bb = lambda ms: np.vstack([m.bounds for m in ms]) * 1e3  # noqa: E731
    allb, nwb = bb(everything), bb(non_wheel)
    wheels_bb = bb([m for n, m in parts.items() if n.startswith("wheel")]) if any(
        n.startswith("wheel") for n in parts) else allb

    r = {}
    width = 2 * allb[:, 1].max()
    r["T3.4_width_min"] = _entry(width - REGS["T3.4_width_min"], f"{width:.1f} mm")
    r["T3.4_width_max"] = _entry(REGS["T3.4_width_max"] - width, f"{width:.1f} mm")
    top = allb[:, 2].max()
    r["T3.5_height"] = _entry(REGS["T3.5_height"] - top, f"{top:.1f} mm")
    lows = {"body": body.bounds[0, 2] * 1e3}
    lows.update({n: m.bounds[0, 2] * 1e3 for n, m in parts.items() if not n.startswith("wheel")})
    low_part = min(lows, key=lows.get)
    clr = lows[low_part]
    r["T3.7_clearance"] = _entry(clr - REGS["T3.7_clearance"],
                                 f"lowest non-wheel part: {low_part} at {clr:.2f} mm")
    r["T7.3_W_min"] = _entry(W - REGS["T7.3_W_min"])
    r["T7.3_W_max"] = _entry(REGS["T7.3_W_max"] - W)
    front = nwb[:, 0].min()
    r["T8.2_nose_overhang"] = _entry(REGS["T8.2_nose_overhang"] - (ref_a - front),
                                     f"{ref_a - front:.1f} mm ahead of Ref A")
    rear = allb[:, 0].max()
    r["T9.4.2_rear_overhang"] = _entry(REGS["T9.4.2_rear_overhang"] - (rear - ref_b),
                                       f"{rear - ref_b:.2f} mm behind Ref B")

    # Nose: body material forward of Ref A (the printed nose) and the wing mount.
    v = body.vertices * 1e3
    nose = v[v[:, 0] < ref_a - 0.5]
    if len(nose):
        r["T8.5.1_nose_z"] = _entry(REGS["T8.5.1_nose_z"] - nose[:, 2].max())
        r["T8.5.1_nose_half_width"] = _entry(REGS["T8.5.1_nose_half_width"] - nose[:, 1].max())

    # Model Block: milled body is everything aft of Ref A.
    milled = v[v[:, 0] >= ref_a - 0.5]
    L = milled[:, 0].max() - milled[:, 0].min()
    Wd = 2 * milled[:, 1].max()
    H = milled[:, 2].max() - milled[:, 2].min()
    r["block_length"] = _entry(REGS["block_length"] - L, f"{L:.1f} mm")
    r["block_width"] = _entry(REGS["block_width"] - Wd, f"{Wd:.1f} mm")
    r["block_height"] = _entry(REGS["block_height"] - H, f"{H:.1f} mm")

    # T4.1 one piece.
    n_mesh = len(body.split(only_watertight=False))
    n = field_bodies if field_bodies is not None else n_mesh
    r["T4.1_single_body"] = _entry(0.0 if n == 1 else -1.0, f"{n} bodies")

    # T5.5 chamber wall (probe points inside the body).
    frac = t55_wall_fraction(body, rear_face_mm=milled[:, 0].max())
    r["T5.5_wall"] = _entry(0.0 if frac >= 0.999 else -(1 - frac) * REGS["T5.5_wall"],
                            f"{100*frac:.1f}% of the 3 mm annulus is solid")

    if mass_state is not None:
        comp = mass_state["competition_mass_g"]
        r["T3.6_mass"] = _entry(comp - REGS["T3.6_mass_g"],
                                f"{comp:.2f} g incl. {mass_state.get('ballast_g', 0):.2f} g ballast")
        if "capacity_g" in mass_state:
            r["T1.22_ballast_capacity"] = _entry(mass_state["capacity_g"] - mass_state["ballast_g"])

    for k, m in assembly.get("gates", {}).items():
        r[f"P4.{k}"] = _entry(m)
    return r


def summary(report: dict) -> dict:
    failed = {k: v for k, v in report.items() if not v["pass"]}
    return {"n_checks": len(report), "n_failed": len(failed), "failed": sorted(failed)}
