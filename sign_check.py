"""
sign_check.py -- does one aero-only level-set step along the adjoint lower drag?

    python sign_check.py --out results/ [--res coarse] [--trust-mm 1.0]

Base car -> one adjoint (unit weight, so the field is dD/dSurface) -> the
production update (apply_adjoint_to_unified, w_mass = 0) taken with +sens and
with -sens -> three forward solves on the same parts. "+" is the pipeline's
descent; it must give the lowest drag. The descent sign in phi_updater has only
ever been checked inside run-to-run noise (2026-07-27, -5.4 % at Re ~ 30).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import paths  # noqa: F401
import run_car as rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="coarse")
    ap.add_argument("--trust-mm", type=float, default=1.0)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--keep-runs", action="store_true")
    a = ap.parse_args(argv)
    # run_car's defaults for everything else.
    ra = argparse.Namespace(**dict(
        W=120.3, x_front=46.0, d_halo=43.72, stage1_mm=2.0, stage1_iters=100, cfd_mm=1.0,
        wheels="carbon_rim_capped", ballast="lead", res=a.res, np=a.np, cfd_iters=2000,
        adj_iters=1000, substeps=a.substeps, trust_mm=a.trust_mm, keep_runs=a.keep_runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import assembly as p4
    from optimizer_contract import GradientWeights

    rc.log("body + parts")
    geom, _ = rc.build_body(ra, out)
    rc.export_body(geom, out / "body_half.stl")
    asm = p4.build(ra.W, ra.x_front, ra.d_halo, str(out / "body_half.stl"), str(out / "parts"),
                   wheel_design=ra.wheels)
    asm["_dir"] = str(out / "parts")
    b = rc.make_bindings(ra, out, asm, seed_geom=geom)
    geom = b.initialize_phi_fields(ra.W, ra.x_front, ra.d_halo, 0)   # the optimiser's start
    m = b.compute_mass_report(geom)
    grads = {"dT_dmass": 0.0, "dT_dh_com": 0.0, "dT_dx_com": 0.0}
    aero_only = GradientWeights(1.0, 0.0, 0.0, 0.0)

    def solve(g, tag):
        gate = b.run_quality_gates(g, tag, str(out / "gates"))
        if not gate.stl_half_path:
            raise SystemExit(f"{tag}: {gate.failure_reason}")
        cfd = b.run_cfd(gate.stl_half_path)
        parts = {k: 2 * v["D_half_N"] for k, v in (cfd.patch_forces or {}).items()}
        rc.log(f"{tag}: D20 {cfd.D20:.5f} N  converged={cfd.converged}  parts {parts}")
        return gate, {"D20": cfd.D20, "stderr": cfd.force_mean_stderr,
                      "converged": cfd.converged, "parts": parts}

    gate, R = solve(geom, "base")
    R = {"base": R}
    rc.log("adjoint")
    adj = b.run_adjoint(gate.stl_half_path, 1.0)
    for sign, tag in ((1.0, "descent"), (-1.0, "ascent")):
        g = copy.deepcopy(geom)
        diag = b.update_phi(g, sign * adj.sensitivity, adj.half_mesh, 1.0, aero_only, grads, m)
        _, R[tag] = solve(g, tag)
        R[tag]["step"] = diag
    d0 = R["base"]["D20"]
    R["verdict"] = {
        "descent_pct": 100 * (R["descent"]["D20"] / d0 - 1),
        "ascent_pct": 100 * (R["ascent"]["D20"] / d0 - 1),
        "sign_ok": bool(R["descent"]["D20"] < d0 < R["ascent"]["D20"]),
    }
    (out / "sign_check.json").write_text(json.dumps(R, indent=2, default=str))
    rc.log(f"verdict {R['verdict']}")
    print(json.dumps(R["verdict"]))


if __name__ == "__main__":
    main()
