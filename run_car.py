"""
run_car.py -- one car through all five parts.

    python run_car.py --out results/ [--cfd] [--optimise N] [--final-cfd]

  1. BODY      Part 1: Stage-1 carve at (W, x_front, d_halo) with legal ballast,
               remapped to the CFD spacing; right-half STL.
  2. PARTS     Part 4: wheels, supports, halo, front + rear wing, tether guides,
               as CFD patches, with masses and regulation gates.
  3. LEGAL     Part 5: scrutineer checks on the assembled car.
  4. CFD       Part 2 (via Part 3 bindings): forward solve with every patch,
               drag by part, race time.                               [--cfd]
  5. OPTIMISE  Part 3 inner loop: CFD + adjoint on the whole car, level-set
               update of the body, ballast absorbing mass changes.  [--optimise]
  6. FINAL     re-place the rear wing on the final body, re-check legality,
               optional final CFD, and write report.md + summary.json.

Every stage writes its artefacts to --out; the run never hides a failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import paths  # noqa: F401

PARTS = paths.PARTS


def log(msg: str) -> None:
    print(f"[run_car {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- body
def build_body(a, out: Path):
    from coarse import use_spacing
    use_spacing(a.stage1_mm)
    import bayesian_outer_search as bos
    import unified_phi as up
    res, geom = bos._level2_evaluate_unified(
        a.W, a.x_front, a.d_halo, n_iters=a.stage1_iters,
        output_dir=str(out / "stage1"), eval_id=0, return_geom=True)
    if geom is None:
        raise SystemExit(f"Stage-1 carve failed: {res.lifecycle}")
    use_spacing(a.cfd_mm)
    geom = up.remap_geometry(geom)
    up.enforce_symmetry(geom)
    return geom, {"stage1_mass_g": res.mass_kg * 1e3, "stage1_T_proxy": res.race_time,
                  "spacing_mm": a.cfd_mm}


def export_body(geom, path: Path) -> dict:
    import unified_phi as up
    from scipy.ndimage import label
    half = up.extract_half_surface(geom)
    half.vertices[half.vertices[:, 1] < 0, 1] = 0.0
    half.export(str(path), file_type="stl_ascii")
    _, n = label(geom.phi.grid < 0)
    if not half.is_watertight:
        raise SystemExit(f"body STL {path.name} is not watertight; Part 2 would reject it")
    return {"faces": len(half.faces), "watertight": bool(half.is_watertight),
            "field_bodies": int(n), "t55_cells_protected": getattr(geom, "t55_cells_protected", 0)}


# --------------------------------------------------------------------------- bindings
def make_bindings(a, out: Path, asm: dict, seed_geom):
    from pipeline_interface import unified_bindings
    import assembly as p4
    extra = p4.extra_surfaces_from(str(Path(asm["_dir"]) / "assembly.json"))
    common = {"resolution": a.res, "n_subdomains": a.np, "extra_surfaces": extra,
              "keep_run_dir": a.keep_runs}
    return unified_bindings(
        thrust_csv_path=str(PARTS["part2-simulation"] / "co2_thrust_data.csv"),
        fixed_hardware_kwargs=asm["fixed_hardware_kwargs"],
        out_dir=str(out / "records"),
        cfd_kwargs=dict(common, max_iterations=a.cfd_iters),
        adjoint_kwargs=dict(common, primal_iters=a.adj_iters, adjoint_iters=a.adj_iters),
        seed_geometry=seed_geom, ballast_material=a.ballast,
        hj_max_substeps=a.substeps, hj_trust_radius_m=a.trust_mm / 1000.0,
    )


def mass_state(b, geom) -> dict:
    import ballast as bl
    m = b.compute_mass_report(geom)
    cap = bl.capacity_kg(bl.DEFAULT_MATERIAL) if m.ballast_regime != "none" else 0.0
    return {"total_g": m.total_mass_kg * 1e3, "competition_mass_g": (m.total_mass_kg - 0.023) * 1e3,
            "ballast_g": m.ballast_kg * 1e3, "capacity_g": cap * 1e3,
            "regime": m.ballast_regime, "com_x_mm": m.com_x_m * 1e3, "com_z_mm": m.com_z_m * 1e3,
            "_report": m}


def cfd_and_objective(b, stl: Path, mstate: dict, wheel_moi: float, mu: float) -> dict:
    from optimizer_contract import DEFAULT_ROLLING_MU  # noqa: F401
    t0 = time.time()
    cfd = b.run_cfd(str(stl))
    m = mstate["_report"]
    _, lx, _, lz = m.launch_com()
    obj = b.evaluate_objective(D20=cfd.D20, L=cfd.L, m_total=m.total_mass_kg,
                               h_com=lz, x_com=lx, mu=mu, wheel_moi=wheel_moi)
    parts = {}
    if cfd.patch_forces:
        for k, v in cfd.patch_forces.items():
            parts[k] = {"D_N": 2 * v["D_half_N"], "L_N": 2 * v["L_half_N"]}
    return {"D20_N": cfd.D20, "L_N": cfd.L, "converged": cfd.converged,
            "residual": cfd.residual_final, "stderr": cfd.force_mean_stderr,
            "drift": cfd.force_drift, "parts": parts, "T_raw_s": obj.T_raw,
            "T_pen_s": obj.T_com_penalized, "gradients": obj.gradients,
            "seconds": round(time.time() - t0, 1)}


def what_if(b, mstate, base: dict, mu: float, designs: dict) -> dict:
    """Race-time deltas from the objective itself, not rules of thumb."""
    m = mstate["_report"]
    _, lx, _, lz = m.launch_com()
    D, L = base["D20_N"], base["L_N"]

    def T(D=D, moi=None, mass=m.total_mass_kg, mu_=mu):
        return b.evaluate_objective(D20=D, L=L, m_total=mass, h_com=lz, x_com=lx,
                                    mu=mu_, wheel_moi=moi).T_raw
    ref = T(moi=designs["_current"])
    out = {"drag -10 %": T(D=0.9 * D, moi=designs["_current"]) - ref,
           "mass +1 g": T(moi=designs["_current"], mass=m.total_mass_kg + 1e-3) - ref,
           "mu 0.010 -> 0.020": T(moi=designs["_current"], mu_=0.020) - ref}
    for name, moi in designs.items():
        if not name.startswith("_"):
            out[f"wheels: {name}"] = T(moi=moi) - ref
    return {k: v * 1e3 for k, v in out.items()}      # ms


def recommend(legal: dict, mstate: dict, cfd: dict | None, whatif: dict | None) -> list:
    rec = []
    bad = [k for k, v in legal.items() if not v["pass"]]
    if bad:
        rec.append(f"FIX LEGALITY FIRST: {', '.join(bad)}.")
    if cfd and cfd["parts"]:
        tot = cfd["D20_N"]
        w = cfd["gradients"].get("dT_dD20", 0.0)
        shares = sorted(((v["D_N"] / tot, k, v["D_N"]) for k, v in cfd["parts"].items()),
                        reverse=True)
        wheels = sum(s for s, k, _ in shares if k.startswith("wheel"))
        top = ", ".join(f"{k} {100*s:.0f} % ({1e3*w*d:.0f} ms)" for s, k, d in shares[:4])
        rec.append(f"Drag by part: {top}.")
        if wheels > 0.5:
            rec.append(f"Wheels carry {100*wheels:.0f} % of the drag. Work on wheel flow first: "
                       "front-wing shape and position ahead of the front wheels, body width "
                       "around the rear wheels, closed wheel faces. Body-only changes act on "
                       "the minority of the drag.")
        if not cfd["converged"]:
            rec.append("The CFD solve did not pass the residual/drift gate: treat its drag as "
                       "indicative and re-run longer before ranking on it.")
    if mstate["regime"] == "absorbing":
        rec.append(f"Ballast absorbs {mstate['ballast_g']:.1f} g of {mstate['capacity_g']:.1f} g: "
                   "body volume is free, so the body shape should follow drag only.")
    elif mstate["regime"] == "heavy":
        rec.append("The car is heavier than 48.2 g with no ballast: remove body material "
                   "(floor channels or a slimmer body) until ballast is needed.")
    elif mstate["regime"] == "full":
        rec.append("The ballast capsule is full and the car is still light: add body mass or "
                   "use a denser ballast (tungsten alloy holds ~25 g).")
    if whatif:
        best = min(((v, k) for k, v in whatif.items() if k.startswith("wheels")), default=None)
        if best and best[0] < -1.0:
            rec.append(f"Wheels: switching to '{best[1].split(': ')[1]}' is worth "
                       f"{-best[0]:.1f} ms (objective, current thrust curve).")
    return rec


# --------------------------------------------------------------------------- report
def write_report(out: Path, S: dict) -> None:
    L = [f"# Car report — W {S['scalars']['W']} mm, x_front {S['scalars']['x_front']} mm, "
         f"d_halo {S['scalars']['d_halo']} mm", ""]
    ms = S["mass"]
    L += ["## Mass", "", "| item | value |", "|---|---|",
          f"| competition mass (T3.6) | {ms['competition_mass_g']:.2f} g |",
          f"| ballast ({S['ballast_material']}) | {ms['ballast_g']:.2f} g of {ms['capacity_g']:.1f} g |",
          f"| ballast regime | {ms['regime']} |",
          f"| COM x / z | {ms['com_x_mm']:.1f} / {ms['com_z_mm']:.1f} mm |",
          f"| wheel design | {S['wheels']['design']} (mean I {S['wheels']['mean_I_kg_m2']*1e9:.1f} g·mm²) |",
          ""]
    lg = S["legality"]
    L += [f"## Legality — {lg['summary']['n_checks']} checks, {lg['summary']['n_failed']} failed", "",
          "| rule | margin | pass | note |", "|---|---|---|---|"]
    for k, v in sorted(lg["checks"].items(), key=lambda kv: (kv[1]["pass"], kv[0])):
        L.append(f"| {k} | {v['margin']:+.2f} | {'yes' if v['pass'] else '**NO**'} | {v['note']} |")
    for tag in ("cfd_initial", "cfd_final"):
        c = S.get(tag)
        if not c:
            continue
        L += ["", f"## {tag.replace('_', ' ').title()}", "",
              f"D20 {c['D20_N']:.4f} N, lift {c['L_N']:+.4f} N, converged {c['converged']} "
              f"(residual {c['residual']:.1e}, stderr {100*(c['stderr'] or 0):.2f} %, "
              f"drift {100*(c['drift'] or 0):.2f} %), T_raw {c['T_raw_s']:.4f} s, "
              f"{c['seconds']:.0f} s wall.", "",
              "| part | drag N | share |", "|---|---|---|"]
        for k, v in sorted(c["parts"].items(), key=lambda kv: -kv[1]["D_N"]):
            L.append(f"| {k} | {v['D_N']:.4f} | {100*v['D_N']/c['D20_N']:.1f} % |")
    if S.get("whatif"):
        L += ["", "## What-if (race objective)", "", "| change | Δ race time |", "|---|---|"]
        for k, v in S["whatif"].items():
            L.append(f"| {k} | {v:+.1f} ms |")
    if S.get("optimisation"):
        o = S["optimisation"]
        L += ["", f"## Optimisation — {o['iterations']} iterations, stop: {o['stop_reason']}", "",
              "| iter | state | D20 N | mass g | T_pen s |", "|---|---|---|---|---|"]
        for h in o["history"]:
            L.append(f"| {h['iteration']} | {h['lifecycle_state']} | "
                     f"{h['D20'] if h['D20'] is None else round(h['D20'], 5)} | "
                     f"{h['total_mass_kg'] if h['total_mass_kg'] is None else round(h['total_mass_kg']*1e3, 2)} | "
                     f"{h['T_penalized'] if h['T_penalized'] is None else round(h['T_penalized'], 4)} |")
    L += ["", "## Recommendations", ""] + [f"- {r}" for r in S["recommendations"]]
    L += ["", "_Absolute times use the project's thrust curve, which has ~60 % of a real "
          "cartridge's impulse; differences and rankings are what to read._"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=float, default=120.3)
    ap.add_argument("--x-front", type=float, default=46.0)
    ap.add_argument("--d-halo", type=float, default=43.72)
    ap.add_argument("--stage1-mm", type=float, default=2.0)
    ap.add_argument("--stage1-iters", type=int, default=100)
    ap.add_argument("--cfd-mm", type=float, default=1.0)
    ap.add_argument("--wheels", default="carbon_rim_capped")
    ap.add_argument("--ballast", default="lead")
    ap.add_argument("--cfd", action="store_true")
    ap.add_argument("--optimise", type=int, default=0)
    ap.add_argument("--final-cfd", action="store_true")
    ap.add_argument("--res", default="coarse")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--cfd-iters", type=int, default=2000)
    ap.add_argument("--adj-iters", type=int, default=1000)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--trust-mm", type=float, default=1.0)
    ap.add_argument("--keep-runs", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import assembly as p4
    import legality
    import wheel as wh
    from optimizer_contract import DEFAULT_ROLLING_MU
    mu = DEFAULT_ROLLING_MU

    S = {"scalars": {"W": a.W, "x_front": a.x_front, "d_halo": a.d_halo},
         "ballast_material": a.ballast, "args": vars(a)}
    log("1 BODY: Stage-1 carve with ballast")
    geom, S["body"] = build_body(a, out)
    S["body"].update(export_body(geom, out / "body_half.stl"))

    log("2 PARTS: Part 4 assembly")
    asm = p4.build(a.W, a.x_front, a.d_halo, str(out / "body_half.stl"), str(out / "parts"),
                   wheel_design=a.wheels)
    asm["_dir"] = str(out / "parts")
    S["wheels"] = asm["wheel_design"]
    S["parts_mass_g"] = asm["parts_mass_g"]

    b = make_bindings(a, out, asm, seed_geom=copy.deepcopy(geom))
    ms = mass_state(b, geom)
    S["mass"] = {k: v for k, v in ms.items() if not k.startswith("_")}

    log("3 LEGAL")
    chk = legality.check(str(out / "body_half.stl"), asm, S["mass"], S["body"]["field_bodies"])
    S["legality"] = {"checks": chk, "summary": legality.summary(chk)}
    moi = asm["wheel_moi_kg_m2"]

    if a.cfd:
        log("4 CFD: forward, all patches")
        S["cfd_initial"] = cfd_and_objective(b, out / "body_half.stl", ms, moi, mu)
        designs = {n: wh.mean_inertia_kg_m2(n) for n in wh.DESIGNS}
        designs["_current"] = moi
        S["whatif"] = what_if(b, ms, S["cfd_initial"], mu, designs)

    if a.optimise:
        log(f"5 OPTIMISE: {a.optimise} CFD+adjoint iterations")
        from inner_loop import run_inner_loop
        from optimizer_contract import GradientWeights, OptimizerConfig
        cfg = OptimizerConfig(rtc_validated_against_track_data=False,
                              cfd_pipeline_validated_on_known_geometry=False,
                              mu=mu, wheel_moi_kg_m2=moi, iteration_budget=a.optimise,
                              evolution_interval_iters=a.optimise)
        start = b.initialize_phi_fields(a.W, a.x_front, a.d_halo, 0)
        res = run_inner_loop(b, cfg, "car", a.W, a.x_front, a.d_halo, start,
                             str(out / "records"), GradientWeights(1.0, 1.0, 0.0, 0.0))
        S["optimisation"] = {"iterations": res.iterations_run, "stop_reason": res.stop_reason,
                             "history": [h.__dict__ for h in res.history]}
        geom = res.final_phi_grids
        S["body_final"] = export_body(geom, out / "body_final_half.stl")
        asm = p4.build(a.W, a.x_front, a.d_halo, str(out / "body_final_half.stl"),
                       str(out / "parts_final"), wheel_design=a.wheels)
        asm["_dir"] = str(out / "parts_final")
        ms = mass_state(b, geom)
        S["mass"] = {k: v for k, v in ms.items() if not k.startswith("_")}
        chk = legality.check(str(out / "body_final_half.stl"), asm, S["mass"],
                             S["body_final"]["field_bodies"])
        S["legality"] = {"checks": chk, "summary": legality.summary(chk)}
        if a.final_cfd:
            log("6 FINAL CFD")
            b2 = make_bindings(a, out, asm, seed_geom=None)
            S["cfd_final"] = cfd_and_objective(b2, out / "body_final_half.stl", ms, moi, mu)

    S["recommendations"] = recommend(S["legality"]["checks"], S["mass"],
                                     S.get("cfd_final") or S.get("cfd_initial"), S.get("whatif"))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    write_report(out, S)
    log(f"done: {S['legality']['summary']}")
    return S


if __name__ == "__main__":
    main()
