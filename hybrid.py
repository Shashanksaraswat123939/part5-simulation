"""
hybrid.py -- the hybrid body optimiser's gradient machinery and its check.

The body (Part 1 param_body, hybrid form) is set by a vector of numbers: the
8-station loft, the sidepods, the blend radius and 32 sculpt modes. One adjoint
solve gives the drag sensitivity on the car's surface; this module turns it into
d(drag)/d(number) for all of them at once:

    surface motion   dn_j(v) = -(phi(p + h e_j) - phi(p - h e_j)) / 2h
                     at each surface vertex v -- geometric, no CFD. Vertices
                     where the milled surface is set by a rule mask or the
                     machining step, not by the design (|phi_design| > 0.75 mm),
                     get dn = 0: the design cannot move them.
    drag gradient    dD/dp_j = C * sum_v s_v dn_j(v) A_v
    volume gradient  dV/dp_j = 2 * sum_v dn_j(v) A_v        (both halves)

C carries the adjoint's sign and scale conventions. It is NOT assumed: the
gradient check measures it. Perturb a few sculpt modes +/- and run real CFD,
then fit the measured drag changes against the predictions.

    python hybrid.py plan --out gc/                      # base skin, cases.json (local)
    python hybrid.py eval --plan gc/cases.json --case 3 --out o/   # forward CFD (CI)
    python hybrid.py adjoint --plan gc/cases.json --out o/         # adjoint + gradient (CI)
    python hybrid.py report results/                     # measured vs predicted
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import paths  # noqa: F401

FD_H = 0.25                 # mm (or unitless for p) central-difference step
DESIGN_BAND_MM = 0.75
CHECK_MODES = (5, 6, 25, 26)      # beside/behind the front wheels, and at the rear axle
CHECK_AMP_MM = 2.0


# ------------------------------------------------------------------ vector <-> params
def names(bp) -> list:
    n = [f"st_b{i}" for i in range(8)] + [f"st_zt{i}" for i in range(8)] \
        + [f"st_zb{i}" for i in range(8)] \
        + ["p", "s_b", "s_zt", "s_zb", "s_x0", "s_x1", "s_taper", "s_p", "blend_mm"] \
        + [f"mode{i}" for i in range(len(bp.modes))]
    if bp.f_b > 0:
        n += ["f_b", "f_z", "f_t"]
    return n


def get(bp, name):
    for key in ("st_b", "st_zt", "st_zb", "mode"):
        if name.startswith(key) and name[len(key):].isdigit():
            seq = getattr(bp, "modes" if key == "mode" else key)
            return seq[int(name[len(key):])]
    return getattr(bp, name)


def put(bp, name, value):
    for key in ("st_b", "st_zt", "st_zb", "mode"):
        if name.startswith(key) and name[len(key):].isdigit():
            attr = "modes" if key == "mode" else key
            seq = list(getattr(bp, attr))
            seq[int(name[len(key):])] = float(value)
            return replace(bp, **{attr: tuple(seq)})
    return replace(bp, **{name: float(value)})


# ------------------------------------------------------------------ geometry
def surface_motion(bp, verts_mm, x_ref_a, x_rear):
    """(n_params, n_verts) normal motion per unit parameter, mm/unit."""
    import param_body as pb
    X, Y, Z = (verts_mm[:, i] for i in range(3))
    phi0 = pb.phi_mm(bp, X, Y, Z, x_ref_a, x_rear)
    live = np.abs(phi0) < DESIGN_BAND_MM
    nm = names(bp)
    D = np.zeros((len(nm), len(verts_mm)))
    for j, n in enumerate(nm):
        v = get(bp, n)
        pp = pb.phi_mm(put(bp, n, v + FD_H), X, Y, Z, x_ref_a, x_rear)
        pm = pb.phi_mm(put(bp, n, v - FD_H), X, Y, Z, x_ref_a, x_rear)
        D[j] = -(pp - pm) / (2 * FD_H) * live
    return nm, D, live


def surface_motion_built(bp, verts_mm, W=120.3, x_front=46.0, d_halo=43.72, skin_mm=0.0,
                         steps=None):
    """(n_params, n_verts) normal motion per unit parameter, measured on the
    BUILT body (rule masks + machining included), by one-sided differences of
    the milled signed-distance field. The geometric shortcut above missed up to
    half the motion where machining reshapes the surface (a wider sidepod also
    fills under itself): 0.57 vs 1.28 cm3 per mm, 2026-09-27."""
    import param_body as pb
    from scipy.interpolate import RegularGridInterpolator

    def field(b):
        g = pb.build(W, x_front, d_halo, b, skin_offset_mm=skin_mm)
        r = g.region
        d = g.spacing_m * 1e3
        ax = [(r.origin_m[i] * 1e3 + np.arange(r.shape[i]) * d) for i in range(3)]
        vm = verts_mm.copy()
        vm[:, 1] = np.abs(vm[:, 1])
        f = RegularGridInterpolator(ax, g.phi.grid * 1e3, bounds_error=False, fill_value=None)
        return f(vm)

    nm = names(bp)
    phi0 = field(bp)
    D = np.zeros((len(nm), len(verts_mm)))
    for j, n in enumerate(nm):
        h = (steps or {}).get(n, 0.2 if n in ("p", "s_p") else 1.0)
        D[j] = -(field(put(bp, n, get(bp, n) + h)) - phi0) / h
    return nm, D


def vertex_areas_mm2(mesh):
    a = np.zeros(len(mesh.vertices))
    np.add.at(a, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3.0, 3))
    return a * 1e6


def gradients(bp, sens, half_mesh, x_ref_a, x_rear, built_skin_mm=None):
    """built_skin_mm: measure the surface motion on the built body (accurate,
    ~65 rebuilds); None: the geometric shortcut (fast, up to 2x off)."""
    verts = np.asarray(half_mesh.vertices) * 1e3
    A = vertex_areas_mm2(half_mesh)
    if built_skin_mm is not None:
        nm, D = surface_motion_built(bp, verts, skin_mm=built_skin_mm)
        live = np.abs(D).max(axis=0) > 1e-6
    else:
        nm, D, live = surface_motion(bp, verts, x_ref_a, x_rear)
    dD = D @ (np.asarray(sens) * A)            # raw adjoint units x mm
    dV = 2 * (D @ A) * 1e-3                    # cm3 per unit
    return {"names": nm, "dD_raw": dD.tolist(), "dV_cm3": dV.tolist(),
            "live_fraction": float(np.mean(live)), "method": "built" if built_skin_mm is not None
            else "geometric"}


# ------------------------------------------------------------------ the check
def _args(extra):
    return ["--res", "medium", "--support", "cad"] + extra


def plan(out: Path) -> dict:
    """Base hybrid body, its mass-sized skin (run locally, no CFD), and the
    check cases: base twice, then CHECK_MODES at +/- CHECK_AMP_MM with the skin
    held fixed so the only change is the mode."""
    import param_body as pb
    import run_car as rc
    base = pb.BodyParams().to_hybrid()
    out.mkdir(parents=True, exist_ok=True)
    S = rc.main(["--out", str(out / "base_local"), "--body-json", json.dumps(base.as_dict())]
                + _args([])[2:])
    skin = S["body"]["build"]["skin_offset_mm"]
    cases = [{"id": 0, "tag": "base", "params": base.as_dict()},
             {"id": 1, "tag": "base_repeat", "params": base.as_dict()}]
    for k in CHECK_MODES:
        for sgn in (1, -1):
            bp = put(base, f"mode{k}", sgn * CHECK_AMP_MM)
            cases.append({"id": len(cases), "tag": f"mode{k}{'+' if sgn > 0 else '-'}",
                          "params": bp.as_dict(), "mode": k, "amp": sgn * CHECK_AMP_MM})
    P = {"skin_mm": skin, "cases": cases, "base_legal": S["legality"]["summary"],
         "base_mass_g": S["mass"]["competition_mass_g"]}
    (out / "cases.json").write_text(json.dumps(P, indent=1))
    return P


def evaluate(P: dict, case_id: int, out: Path) -> dict:
    import run_car as rc
    c = next(c for c in P["cases"] if c["id"] == case_id)
    S = rc.main(["--out", str(out), "--cfd", "--body-json", json.dumps(c["params"]),
                 "--skin-mm", str(P["skin_mm"])] + _args([]))
    r = {"id": case_id, "tag": c["tag"], "D20_N": S["cfd_initial"]["D20_N"],
         "converged": S["cfd_initial"]["converged"], "mode": c.get("mode"), "amp": c.get("amp"),
         "mass_g": S["mass"]["competition_mass_g"], "legal": S["legality"]["summary"]}
    (out / f"gc_case_{case_id}.json").write_text(json.dumps(r, indent=1, default=str))
    return r


def adjoint(P: dict, out: Path) -> dict:
    import param_body as pb
    import run_car as rc
    base = pb.BodyParams.from_dict(P["cases"][0]["params"])
    a = argparse.Namespace(
        W=120.3, x_front=46.0, d_halo=43.72, stage1_mm=2.0, stage1_iters=100, cfd_mm=1.0,
        wheels="team_stl", ballast="none", res="medium", np=4, cfd_iters=2000, adj_iters=1000,
        substeps=6, trust_mm=1.0, smooth_mm=0.0, keep_runs=False, support="cad",
        support_json=None, nose_json=None, body_json=json.dumps(base.as_dict()),
        skin_mm=P["skin_mm"])
    out.mkdir(parents=True, exist_ok=True)
    geom, info, asm = rc.build_car(a, out, **rc.part4_kwargs(a))
    b = rc.make_bindings(a, out, asm, seed_geom=None)
    gate = b.run_quality_gates(geom, "adj", str(out / "gates"))
    adj = b.run_adjoint(gate.stl_half_path, 1.0)
    lm = geom.landmarks
    G = gradients(base, adj.sensitivity, adj.half_mesh,
                  lm["ref_plane_A_m"] * 1e3, lm["rear_face_m"] * 1e3, built_skin_mm=P["skin_mm"])
    np.savez_compressed(out / "adjoint.npz", sens=np.asarray(adj.sensitivity),
                        verts=np.asarray(adj.half_mesh.vertices), faces=np.asarray(adj.half_mesh.faces))
    (out / "gradient.json").write_text(json.dumps(G, indent=1))
    return G


def report(root: Path) -> str:
    cases = [json.loads(p.read_text()) for p in root.rglob("gc_case_*.json")]
    G = json.loads(next(root.rglob("gradient.json")).read_text())
    by = {c["tag"]: c for c in cases}
    bases = [by[t]["D20_N"] for t in ("base", "base_repeat") if t in by]
    D0 = float(np.mean(bases))
    lines = [f"base D20 {D0:.5f} N (runs {', '.join(f'{b:.5f}' for b in bases)}; "
             f"spread {100 * (max(bases) - min(bases)) / D0:.2f} %); "
             f"adjoint live surface {100 * G['live_fraction']:.0f} %", "",
             "| case | measured dD N | predicted (raw) | converged |", "|---|---|---|---|"]
    xs, ys = [], []
    idx = {n: i for i, n in enumerate(G["names"])}
    for c in sorted(cases, key=lambda c: c["id"]):
        if c.get("mode") is None:
            continue
        pred = G["dD_raw"][idx[f"mode{c['mode']}"]] * c["amp"]
        meas = c["D20_N"] - D0
        xs.append(pred)
        ys.append(meas)
        lines.append(f"| {c['tag']} | {meas:+.5f} | {pred:+.4e} | {c['converged']} |")
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) >= 2 and np.any(xs):
        C = float(xs @ ys / (xs @ xs))
        resid = ys - C * xs
        r = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 0 and np.std(ys) > 0 else float("nan")
        sign_ok = int(np.sum(np.sign(xs * C) == np.sign(ys)))
        lines += ["", f"fitted scale C = {C:+.4e}  (sign: {'+' if C > 0 else '-'})",
                  f"correlation measured vs predicted r = {r:.2f}; sign agreement "
                  f"{sign_ok}/{len(xs)}; rms residual {np.sqrt(np.mean(resid ** 2)):.5f} N "
                  f"vs noise {abs(bases[0] - bases[-1]) if len(bases) > 1 else float('nan'):.5f} N"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--plan", required=True)
    e.add_argument("--case", type=int, required=True)
    e.add_argument("--out", required=True)
    j = sp.add_parser("adjoint")
    j.add_argument("--plan", required=True)
    j.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("root")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        P = plan(Path(a.out))
        print(json.dumps({k: v for k, v in P.items() if k != "cases"}, default=str),
              len(P["cases"]), "cases")
    elif a.cmd == "eval":
        print(evaluate(json.loads(Path(a.plan).read_text()), a.case, Path(a.out)))
    elif a.cmd == "adjoint":
        G = adjoint(json.loads(Path(a.plan).read_text()), Path(a.out))
        print({k: G[k] for k in ("live_fraction",)})
    else:
        print(report(Path(a.root)))


if __name__ == "__main__":
    main()
