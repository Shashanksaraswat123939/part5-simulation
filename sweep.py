"""
sweep.py -- one wheel/wing variant on a fixed body: legality, CFD, race time.

    python sweep.py --variant fwing_aoa_p6 --out results/ [--res medium]
    python sweep.py --summarise results/            # table of every variant

The body is the same Stage-1 carve every run (deterministic), so variants
differ only in the Part 4 parts. `base` and `base_repeat` are identical: their
gap is the remeshing noise, and anything smaller is not a result.

Race time is the locked objective with the variant's own wheel inertia. The car
sits in the ballast-absorbing regime, so part-mass changes of a few 0.1 g move
ballast, not race time; they are reported, not priced.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import paths  # noqa: F401
import run_car as rc

VARIANTS = {
    "base": {},
    "base_repeat": {},
    # wheels: rounded tyre shoulders beyond the T7.4 contact width, domed hubcap
    "wheel_round_out": {"wheel": dict(shoulder_out=1.5)},
    "wheel_round_both": {"wheel": dict(shoulder_in=1.5, shoulder_out=1.5)},
    "wheel_dome": {"wheel": dict(dome=2.0)},
    "wheel_round_out_dome": {"wheel": dict(shoulder_out=1.5, dome=2.0)},
    # front wing: incidence, height, span, minimum section
    "fwing_aoa_p6": {"front": dict(aoa_deg=6.0)},
    "fwing_aoa_m6": {"front": dict(aoa_deg=-6.0)},
    "fwing_high": {"front": dict(z_chord_mm=16.0)},
    "fwing_wide": {"front": dict(half_span_mm=42.0)},
    "fwing_min": {"front": dict(chord_mm=15.0, t_frac=0.14)},
    # rear wing: minimum legal section
    "rwing_min": {"rear": dict(chord_mm=15.0, t_frac=0.14)},
    # round 2 (2026-09-26): +6 deg cut the front-wheel drag 10 %; map the
    # incidence, re-run the unconverged rear wing, and combine the winners
    "base_repeat2": {},
    "fwing_aoa_p3": {"front": dict(aoa_deg=3.0)},
    "fwing_aoa_p9": {"front": dict(aoa_deg=9.0)},
    "fwing_aoa_p12": {"front": dict(aoa_deg=12.0, z_chord_mm=9.0)},   # 8 mm breaks T8.7
    "fwing_min_aoa_p6": {"front": dict(chord_mm=15.0, t_frac=0.14, aoa_deg=6.0)},
    "rwing_min_repeat": {"rear": dict(chord_mm=15.0, t_frac=0.14)},
    "combo_p6": {"front": dict(chord_mm=15.0, t_frac=0.14, aoa_deg=6.0),
                 "rear": dict(chord_mm=15.0, t_frac=0.14), "wheel": dict(dome=2.0)},
    "combo_p9": {"front": dict(chord_mm=15.0, t_frac=0.14, aoa_deg=9.0),
                 "rear": dict(chord_mm=15.0, t_frac=0.14), "wheel": dict(dome=2.0)},
    # round 3: base is now the round-2 wing winners (Part 4 defaults); does
    # the 30 mm nose (-7.4 % on the old wings) still pay, at a 0.5 mm wall
    # that fits inside the ~1.5 g of ballast the lighter wings freed?
    "nose_30_thin": {"nose": dict(length_mm=30.0, wall_mm=0.5)},
    "nose_30_thin_repeat": {"nose": dict(length_mm=30.0, wall_mm=0.5)},
    # round 4 (base = round-2 winners). The rear-wing gain was measured on
    # unconverged solves: re-run old vs new at 4000 iterations.
    "base_4k": {"cfd_iters": 4000},
    "base_4k_repeat": {"cfd_iters": 4000},
    "rwing_old_4k": {"rear": dict(chord_mm=16.0, t_frac=0.15), "cfd_iters": 4000},
    "rwing_old_4k_repeat": {"rear": dict(chord_mm=16.0, t_frac=0.15), "cfd_iters": 4000},
    # hubcap dome size (2 mm gave -0.7 % on the old wings)
    "wheel_dome_1": {"wheel": dict(dome=1.0)},
    "wheel_dome_3": {"wheel": dict(dome=3.0)},
    "wheel_dome_4": {"wheel": dict(dome=4.0)},
    # front wing at the new +6 deg / minimum section: span, height, gap, flap
    "fw6_wide": {"front": dict(half_span_mm=42.0)},
    "fw6_low": {"front": dict(z_chord_mm=7.0)},
    "fw6_high": {"front": dict(z_chord_mm=10.0)},
    "fw6_gap8": {"front": dict(gap_to_wheel_mm=8.0)},
    "fw6_flap20": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=20.0)},
    "fw6_flap30": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0)},
    # round 5, the final car: round-4 winners (flap at 30 deg, 1 mm dome,
    # and the OLD rear wing, which beat the minimum one once both converged)
    "dome1_repeat": {"wheel": dict(dome=1.0)},
    "flap30_repeat": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0)},
    "final": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0), "wheel": dict(dome=1.0),
              "rear": dict(chord_mm=16.0, t_frac=0.15)},
    "final_repeat": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0), "wheel": dict(dome=1.0),
                     "rear": dict(chord_mm=16.0, t_frac=0.15)},
    "final_repeat2": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0), "wheel": dict(dome=1.0),
                      "rear": dict(chord_mm=16.0, t_frac=0.15)},
    "final_nodome": {"front": dict(flap_chord_mm=8.0, flap_aoa_deg=30.0),
                     "rear": dict(chord_mm=16.0, t_frac=0.15)},
    # printed nose cone ahead of Ref A (Part 4 nose.py)
    "nose_20": {"nose": dict()},
    "nose_30": {"nose": dict(length_mm=30.0)},
    "nose_20_sharp": {"nose": dict(k=0.75)},
    "nose_20_droop": {"nose": dict(tip_z_mm=7.0)},
    "nose_20_boxy": {"nose": dict(p=4.0)},
}


def run_variant(name: str, out: Path, res: str, np_: int, keep_runs: bool) -> dict:
    import assembly as p4
    import legality
    import nose as ns
    import wheel as wh
    import wings as wg
    from optimizer_contract import DEFAULT_ROLLING_MU
    v = VARIANTS[name]
    a = argparse.Namespace(
        W=120.3, x_front=46.0, d_halo=43.72, stage1_mm=2.0, stage1_iters=100, cfd_mm=1.0,
        wheels="carbon_rim_capped", ballast="lead", res=res, np=np_,
        cfd_iters=v.get("cfd_iters", 2000),
        adj_iters=1000, substeps=6, trust_mm=1.0, smooth_mm=0.0, keep_runs=keep_runs)
    out.mkdir(parents=True, exist_ok=True)
    geom, _ = rc.build_body(a, out)
    body = rc.export_body(geom, out / "body_half.stl")
    f, r = wh.design(a.wheels)
    wheels = (replace(f, **v.get("wheel", {})), replace(r, **v.get("wheel", {})))
    asm = p4.build(a.W, a.x_front, a.d_halo, str(out / "body_half.stl"), str(out / "parts"),
                   wheel_design=wheels, front=replace(wg.FrontWing(), **v.get("front", {})),
                   rear=replace(wg.RearWing(), **v.get("rear", {})),
                   nose=replace(ns.NoseCone(), **v["nose"]) if "nose" in v else None)
    asm["_dir"] = str(out / "parts")
    b = rc.make_bindings(a, out, asm, seed_geom=None)
    ms = rc.mass_state(b, geom)
    chk = legality.check(str(out / "body_half.stl"), asm,
                         {k: x for k, x in ms.items() if not k.startswith("_")},
                         body["field_bodies"])
    moi = asm["wheel_moi_kg_m2"]
    cfd = rc.cfd_and_objective(b, out / "body_half.stl", ms, moi, DEFAULT_ROLLING_MU)
    R = {"variant": name, "spec": v, "D20_N": cfd["D20_N"], "L_N": cfd["L_N"],
         "T_raw_s": cfd["T_raw_s"], "converged": cfd["converged"], "stderr": cfd["stderr"],
         "parts": cfd["parts"], "wheel_moi_kg_m2": moi,
         "wheel_mass_g": [wheels[0].mass, wheels[1].mass], "parts_mass_g": asm["parts_mass_g"],
         "legality": legality.summary(chk)}
    (out / f"sweep_{name}.json").write_text(json.dumps(R, indent=2, default=str))
    rc.log(f"{name}: D20 {R['D20_N']:.5f} N  T {R['T_raw_s']:.5f} s  {R['legality']}")
    return R


def summarise(root: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(root.rglob("sweep_*.json"))]
    by = {r["variant"]: r for r in rows}
    bases = [by[k] for k in ("base", "base_repeat", "base_repeat2") if k in by]
    if not bases:
        return "no base run"
    D0 = sum(x["D20_N"] for x in bases) / len(bases)
    T0 = sum(x["T_raw_s"] for x in bases) / len(bases)
    ds = [x["D20_N"] for x in bases]
    noise = (max(ds) - min(ds)) / D0 * 100 if len(ds) > 1 else float("nan")
    out = [f"base D20 {D0:.4f} N, T {T0:.4f} s; spread of {len(ds)} identical base runs "
           f"{noise:.2f} % (the noise floor). dT includes part mass: the base car is "
           f"above the 48 g target, so grams count here; after ballast they would not.", "",
           "| variant | D20 N | dD20 % | dT ms | wheel I g.mm2 | legal | wheels / wings+nose drag N |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["T_raw_s"]):
        p = r["parts"]
        wd = sum(p.get(k, {}).get("D_N", 0) for k in ("wheelF", "wheelR"))
        gd = sum(p.get(k, {}).get("D_N", 0) for k in ("fwing", "rwing", "nose"))
        leg = "yes" if r["legality"]["n_failed"] == 0 else ",".join(r["legality"]["failed"])
        out.append(f"| {r['variant']} | {r['D20_N']:.4f} | {100 * (r['D20_N'] / D0 - 1):+.2f} | "
                   f"{1e3 * (r['T_raw_s'] - T0):+.2f} | {r['wheel_moi_kg_m2'] * 1e9:.1f} | {leg} | "
                   f"{wd:.4f} / {gd:.4f} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--out")
    ap.add_argument("--res", default="medium")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--summarise")
    a = ap.parse_args(argv)
    if a.summarise:
        text = summarise(Path(a.summarise))
        print(text)
        return text
    return run_variant(a.variant, Path(a.out), a.res, a.np, a.keep_runs)


if __name__ == "__main__":
    main()
