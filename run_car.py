"""
run_car.py -- one car through all five parts.

    python run_car.py --out results/ [--body-json '{...}'] [--cfd] [--optimise N] [--final-cfd]

  1. BODY      Part 1: the parametric body (param_body), machinable, sized so
               the car AS MANUFACTURED weighs 48.2 g (no ballast); right-half STL.
  2. PARTS     Part 4: wheels, supports, halo, nose, front + rear wing, tether
               guides as CFD patches; pockets and printed parts (joints); the
               manufacturing files; masses and regulation gates.
  3. LEGAL     Part 5: scrutineer checks on the assembled car.
  4. CFD       Part 2 (via Part 3 bindings): forward solve with every patch,
               drag by part, race time.                               [--cfd]
  5. OPTIMISE  Part 3 inner loop: CFD + adjoint on the whole car, level-set
               update of the body.                                  [--optimise]
  6. FINAL     re-place the rear wing on the final body, re-check legality,
               optional final CFD, and write report.md + summary.json.

Every stage writes its artefacts to --out; the run never hides a failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import paths  # noqa: F401

PARTS = paths.PARTS


def log(msg: str) -> None:
    print(f"[run_car {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- body
def build_body(a, out: Path):
    """The parametric body (Part 1 param_body): drawn, made machinable, and
    sized by a skin offset (or at a fixed skin with --skin-mm)."""
    from coarse import use_spacing
    import param_body as pb
    use_spacing(a.cfd_mm)
    bp = pb.BodyParams.from_dict(json.loads(a.body_json))
    g = pb.build(a.W, a.x_front, a.d_halo, bp,
                 target_body_kg=getattr(a, "_body_target_kg", None),
                 skin_offset_mm=getattr(a, "skin_mm", None))
    return g, {"param_body": bp.as_dict(), "spacing_mm": a.cfd_mm,
               "build": getattr(g, "build_report", {})}


def parts_and_joints(a, out: Path, geom, body_stl: Path, parts_dir: str, **p4kw) -> dict:
    """Part 4 parts on this body, then the joints: every printed part becomes a
    positive glued into a milled pocket (Part 4 joints.py). The mass the
    joints move (plastic plugs for foam, buried CAD trimmed off) goes into the
    hardware masses the rollup uses. Manufacturing files go to out/manufacture."""
    import assembly as p4
    import joints
    asm = p4.build(a.W, a.x_front, a.d_halo, str(body_stl), str(out / parts_dir), **p4kw)
    asm["_dir"] = str(out / parts_dir)
    import trimesh
    lm = geom.landmarks
    jr = joints.make_all(trimesh.load(str(body_stl), force="mesh"), asm,
                         lm["ref_plane_A_m"] * 1e3, lm["rear_face_m"] * 1e3,
                         out / "manufacture")
    f, d = asm["fixed_hardware_kwargs"], jr["mass_delta_kg"]
    sup = d.get("supports", 0.0)
    f["wheels_front_mass_kg"] += sup / 2
    f["wheels_rear_mass_kg"] += sup / 2
    f["wheels_axles_mass_kg"] += sup
    f["rear_wing_mass_kg"] += sum(v for k, v in d.items() if k != "supports")
    asm["joints"] = jr
    return asm


def manufactured_mass_kg(asm) -> float:
    """The car as it is made, without the cartridge: every machined and printed
    file's own volume x its material (Part 4 joints), the wheels with their
    bearings, and the halo. This, not the model's rollup, is what the
    scrutineer's scale reads: on 2026-09-30 a car the rollup put at 48.21 g
    was 47.84 g as manufactured -- under the 48.0 g floor."""
    wd = asm["wheel_design"]
    return (asm["joints"]["manufactured_g"] + 2 * wd["front"]["mass_g"] + 2 * wd["rear"]["mass_g"]
            ) * 1e-3 + asm["fixed_hardware_kwargs"]["halo_mass_kg"]


def build_car(a, out: Path, **p4kw):
    """Body and parts to a fixed point: the body is sized so the car AS
    MANUFACTURED weighs the target, but the joints (pockets and plugs) and so
    the parts' masses depend on the body."""
    import ballast
    import unified_phi as up
    target = None
    targets = []
    for pass_ in range(1, 7):
        prev = target
        a._body_target_kg = target
        geom, info = build_body(a, out)
        info.update(export_body(geom, out / "body_half.stl"))
        asm = parts_and_joints(a, out, geom, out / "body_half.stl", "parts", **p4kw)
        # ask the body for the shortfall on top of what it weighs now
        new = (sum(c.mass_kg for c in up.compute_mass_com(geom))
               + ballast.target_competition_kg() - manufactured_mass_kg(asm))
        # damped: the keel under a support beam grows as the body grows, so an
        # undamped fixed point overshoots and oscillates (1.6 g off, 2026-09-27)
        target = new if prev is None else 0.5 * (prev + new)
        targets.append(new * 1e3)
        log(f"  pass {pass_}: body target {new*1e3:.2f} g; joints "
            + ", ".join(f"{k} {v*1e3:+.2f}" for k, v in asm["joints"]["mass_delta_kg"].items())
            + f"; supports each {[round(x, 2) for x in asm['info'].get('support_mass_g_each', [])]}")
        if getattr(a, "skin_mm", None) is not None:
            break                                  # fixed skin: nothing to iterate
        if pass_ >= 2 and abs(new - prev) < 5e-5:
            break
    info["body_target_g"] = target * 1e3
    info["build_passes"] = pass_
    info["body_targets_g"] = targets
    return geom, info, asm


def part4_kwargs(a) -> dict:
    import beam_support as bsm
    kind = getattr(a, "support", "beam")
    sup = {"cad": None, "beam": bsm.BeamSupport()}[kind]
    if kind == "beam" and getattr(a, "support_json", None):
        sup = bsm.BeamSupport(**json.loads(a.support_json))
    # The body ends at Ref A; the nose cone is a printed SLS PA12 shell (team
    # spec 2026-09-27) whose root takes the body's section 1 mm aft of Ref A.
    import nose as ns
    nk = dict(blend_after_ref_a_mm=1.0, root_scale=1.0, material="PA12", wall_mm=0.8)
    if getattr(a, "nose_json", None):
        nk.update(json.loads(a.nose_json))
    kw = {"support": sup, "nose": ns.NoseCone(**nk)}
    if getattr(a, "fwing_json", None):
        import wings as wg
        kw["front"] = wg.FrontWing(**json.loads(a.fwing_json))
    return kw


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
    # ONE meshing frame for every car: the legal envelope (T8.2 and T9.4.2
    # overhangs, T3.4 half width, T3.5 height, 1 mm slack). Sized from each
    # car's own bounding box, a longer nose or a wider wing moved the domain,
    # the background cells and the refinement boxes -- remeshing noise that
    # has nothing to do with the change being measured.
    ref_a, ref_b = a.x_front - 16.0, a.x_front + a.W + 16.0
    frame = (((ref_a - 41.0) / 1e3, 0.0, 0.0), ((ref_b + 41.0) / 1e3, 0.0435, 0.066))
    common = {"resolution": a.res, "n_subdomains": a.np, "extra_surfaces": extra,
              "keep_run_dir": a.keep_runs, "domain_reference_bounds": frame}
    cfd_extra = ({"stage_timeout_s": a.stage_timeout_s}
                 if getattr(a, "stage_timeout_s", None) else {})
    cfd_extra.update(json.loads(getattr(a, "cfd_json", None) or "{}"))
    return unified_bindings(
        thrust_csv_path=str(PARTS["part2-simulation"] / "co2_thrust_data.csv"),
        fixed_hardware_kwargs=asm["fixed_hardware_kwargs"],
        out_dir=str(out / "records"),
        cfd_kwargs=dict(common, max_iterations=a.cfd_iters, **cfd_extra),
        adjoint_kwargs=dict(common, primal_iters=a.adj_iters, adjoint_iters=a.adj_iters),
        seed_geometry=seed_geom,
        ballast_material=None,               # the team adds no ballast
        hj_max_substeps=a.substeps, hj_trust_radius_m=a.trust_mm / 1000.0,
        hj_aero_smooth_m=a.smooth_mm / 1000.0,
    )


def mass_state(b, geom) -> dict:
    m = b.compute_mass_report(geom)
    return {"total_g": m.total_mass_kg * 1e3, "competition_mass_g": (m.total_mass_kg - 0.023) * 1e3,
            "com_x_mm": m.com_x_m * 1e3, "com_z_mm": m.com_z_m * 1e3, "_report": m}


def cfd_and_objective(b, stl: Path, mstate: dict, wheel_moi: float, mu: float) -> dict:
    import ballast as _bl
    t0 = time.time()
    cfd = b.run_cfd(str(stl))
    m = mstate["_report"]
    _, lx, _, lz = m.launch_com()
    m_car = (mstate["manufactured_mass_g"] * 1e-3 + _bl.CARTRIDGE_KG
             if "manufactured_mass_g" in mstate else m.total_mass_kg)
    obj = b.evaluate_objective(D20=cfd.D20, L=cfd.L, m_total=m_car,
                               h_com=lz, x_com=lx, mu=mu, wheel_moi=wheel_moi)
    # The same car at exactly the mass target: what a body change is worth once
    # re-sized, so fixed-skin screening cars compare fairly (pattern.py).
    m_target = _bl.target_competition_kg() + _bl.CARTRIDGE_KG
    obj_t = b.evaluate_objective(D20=cfd.D20, L=cfd.L, m_total=m_target,
                                 h_com=lz, x_com=lx, mu=mu, wheel_moi=wheel_moi)
    parts = {}
    if cfd.patch_forces:
        for k, v in cfd.patch_forces.items():
            parts[k] = {"D_N": 2 * v["D_half_N"], "L_N": 2 * v["L_half_N"],
                        "y_plus": v.get("y_plus")}
    return {"D20_N": cfd.D20, "L_N": cfd.L, "converged": cfd.converged,
            "residual": cfd.residual_final, "stderr": cfd.force_mean_stderr,
            "drift": cfd.force_drift, "parts": parts, "T_raw_s": obj.T_raw,
            "T_at_target_s": obj_t.T_raw,
            "T_pen_s": obj.T_com_penalized, "gradients": obj.gradients,
            "seconds": round(time.time() - t0, 1)}


def what_if(b, mstate, base: dict, mu: float, moi: float) -> dict:
    """Race-time deltas from the objective itself, not rules of thumb (ms)."""
    m = mstate["_report"]
    _, lx, _, lz = m.launch_com()
    D, L = base["D20_N"], base["L_N"]

    def T(D=D, mass=m.total_mass_kg, mu_=mu):
        return b.evaluate_objective(D20=D, L=L, m_total=mass, h_com=lz, x_com=lx,
                                    mu=mu_, wheel_moi=moi).T_raw
    ref = T()
    return {k: (v - ref) * 1e3 for k, v in {
        "drag -10 %": T(D=0.9 * D), "mass +1 g": T(mass=m.total_mass_kg + 1e-3),
        "mu 0.010 -> 0.020": T(mu_=0.020)}.items()}


def recommend(legal: dict, cfd: dict | None) -> list:
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
    return rec


# --------------------------------------------------------------------------- report
def write_report(out: Path, S: dict) -> None:
    L = [f"# Car report — W {S['scalars']['W']} mm, x_front {S['scalars']['x_front']} mm, "
         f"d_halo {S['scalars']['d_halo']} mm", ""]
    ms = S["mass"]
    L += ["## Mass", "", "| item | value |", "|---|---|",
          f"| mass as manufactured (T3.6) | {ms.get('manufactured_mass_g', ms['competition_mass_g']):.2f} g |",
          f"| mass by the model's rollup | {ms['competition_mass_g']:.2f} g |",
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
    ap.add_argument("--cfd-mm", type=float, default=1.0)
    ap.add_argument("--support", choices=("cad", "beam"), default="beam",
                    help="beam = the team's architecture, parametric; cad = the v2 CAD supports (reference)")
    ap.add_argument("--support-json", default=None, help="BeamSupport parameters as JSON")
    ap.add_argument("--nose-json", default=None, help="NoseCone parameters as JSON (parametric body)")
    ap.add_argument("--fwing-json", default=None, help="FrontWing parameters as JSON")
    ap.add_argument("--skin-mm", type=float, default=None,
                    help="fixed skin offset: no mass sizing (gradient checks)")
    ap.add_argument("--body-json", default="{}",
                    help="parametric body (Part 1 param_body.BodyParams) as JSON; '{}' = the start car")
    ap.add_argument("--cfd", action="store_true")
    ap.add_argument("--optimise", type=int, default=0)
    ap.add_argument("--final-cfd", action="store_true")
    ap.add_argument("--res", default="coarse")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--cfd-iters", type=int, default=2000)
    ap.add_argument("--adj-iters", type=int, default=1000)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--trust-mm", type=float, default=1.0)
    ap.add_argument("--smooth-mm", type=float, default=0.0)
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--stage-timeout-s", type=int, default=None,
                    help="per OpenFOAM stage; fine meshes need more than the 7200 s default")
    ap.add_argument("--cfd-json", default=None,
                    help='extra Part 2 run_half_car_cfd options, e.g. \'{"turbulence_model": "kOmegaSSTLM"}\'')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import assembly as p4
    import legality
    from optimizer_contract import DEFAULT_ROLLING_MU
    mu = DEFAULT_ROLLING_MU

    S = {"scalars": {"W": a.W, "x_front": a.x_front, "d_halo": a.d_halo},
         "args": vars(a)}
    log("1-2 BODY + PARTS: parametric body sized against the manufactured parts")
    p4kw = part4_kwargs(a)
    geom, S["body"], asm = build_car(a, out, **dict(p4kw))
    S["wheels"] = asm["wheel_design"]
    S["parts_mass_g"] = asm["parts_mass_g"]

    b = make_bindings(a, out, asm, seed_geom=copy.deepcopy(geom))
    ms = mass_state(b, geom)
    ms["manufactured_mass_g"] = manufactured_mass_kg(asm) * 1e3
    S["mass"] = {k: v for k, v in ms.items() if not k.startswith("_")}
    S["manufactured"] = asm["joints"]["manufactured"]

    log("3 LEGAL")
    chk = legality.check(str(out / "body_half.stl"), asm, S["mass"], S["body"]["field_bodies"],
                         machined_stl=asm["joints"].get("machined_body_half_stl"))
    S["legality"] = {"checks": chk, "summary": legality.summary(chk)}
    moi = asm["wheel_moi_kg_m2"]

    if a.cfd:
        log("4 CFD: forward, all patches")
        # The same decimated STL the optimiser's gate would mesh.
        gate = b.run_quality_gates(copy.deepcopy(geom), "initial", str(out / "gates"))
        if not gate.stl_half_path:
            raise SystemExit(f"initial body failed the CFD gate: {gate.failure_reason}")
        S["cfd_initial"] = cfd_and_objective(b, Path(gate.stl_half_path), ms, moi, mu)
        S["whatif"] = what_if(b, ms, S["cfd_initial"], mu, moi)

    if a.optimise:
        log(f"5 OPTIMISE: {a.optimise} CFD+adjoint iterations")
        from inner_loop import run_inner_loop
        from optimizer_contract import GradientWeights, OptimizerConfig
        cfg = OptimizerConfig(rtc_validated_against_track_data=False,
                              cfd_pipeline_validated_on_known_geometry=False,
                              mu=mu, wheel_moi_kg_m2=moi, iteration_budget=a.optimise,
                              evolution_interval_iters=a.optimise)
        start = copy.deepcopy(geom)
        res = run_inner_loop(b, cfg, "car", a.W, a.x_front, a.d_halo, start,
                             str(out / "records"), GradientWeights(1.0, 1.0, 0.0, 0.0))
        S["optimisation"] = {"iterations": res.iterations_run, "stop_reason": res.stop_reason,
                             "history": [h.__dict__ for h in res.history]}
        # The BEST measured iterate, not the loop's last state: that one is the
        # geometry after the final update, which no CFD ever saw. On 2026-09-26
        # every aero-only step raised D20, so "last" was reliably the worst.
        geom = res.final_phi_grids
        if res.best is not None and res.best.phi_snapshot_paths:
            geom.phi.load(next(iter(res.best.phi_snapshot_paths.values())))
            S["optimisation"]["final_is"] = res.best.candidate_id
        S["body_final"] = export_body(geom, out / "body_final_half.stl")
        asm = parts_and_joints(a, out, geom, out / "body_final_half.stl", "parts_final",
                               **p4kw)
        ms = mass_state(b, geom)
        ms["manufactured_mass_g"] = manufactured_mass_kg(asm) * 1e3
        S["mass"] = {k: v for k, v in ms.items() if not k.startswith("_")}
        chk = legality.check(str(out / "body_final_half.stl"), asm, S["mass"],
                             S["body_final"]["field_bodies"],
                             machined_stl=asm["joints"].get("machined_body_half_stl"))
        S["legality"] = {"checks": chk, "summary": legality.summary(chk)}
        if a.final_cfd:
            log("6 FINAL CFD")
            b2 = make_bindings(a, out, asm, seed_geom=None)
            S["cfd_final"] = cfd_and_objective(b2, out / "body_final_half.stl", ms, moi, mu)

    S["recommendations"] = recommend(S["legality"]["checks"],
                                     S.get("cfd_final") or S.get("cfd_initial"))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    write_report(out, S)
    log(f"done: {S['legality']['summary']}")
    return S


if __name__ == "__main__":
    main()
