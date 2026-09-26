"""
theory.py -- how fast can a legal car be? Race time for measured and limiting
cars, from the same locked race objective the optimiser uses.

    python theory.py [--summary results/summary.json]

Two thrust scales: the project curve (x1.0, what every other number here uses)
and x1.5, which reproduced the 2022 World Finals fastest car (1.05 s predicted
vs 1.06 s measured; rnd/06_RND_RESULTS.md). Rankings agree; absolute times
should be read on the x1.5 column.

Limits are physical or regulatory, not design guesses:
  * mass     T3.6 floor, 48.0 g + 0.2 g scale margin, plus the 23 g cartridge
  * wheels   4 exposed rotating wheels at the T7.4 / T7.5 minimum size cannot
             be faired (T7.9-T7.11): their drag is the car's drag floor
  * inertia  a wheel cannot spin with zero inertia; the 0.20 mm carbon rim
             alone is the lightest structure measured (rnd/wheels)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import paths  # noqa: F401

M_TOTAL = 0.0482 + 0.023            # T3.6 target + cartridge, kg

# The measured final car (GitHub Actions, medium mesh, 2026-09-26/27).
FINAL = dict(D20=0.3881, L=0.254, I=124.9e-9, mu=0.010, h=0.030)   # launch COM; its penalty is a placeholder polynomial centred on 30 mm
PARTS_N = {"wheelF": 0.1616, "wheelR": 0.0955, "car": 0.0465, "supports": 0.0254,
           "halo": 0.0177, "fwing": 0.0256, "rwing": 0.0135, "tethers": 0.0024}


def race_time(model, D20, L, I, mu, h, m=M_TOTAL, thrust=1.0):
    import jax.numpy as jnp
    from race_objective import race_time_seconds
    mdl = model._replace(force_coeffs=model.force_coeffs * thrust)
    p = jnp.array([D20, m, mu, I, 1.0, h, L, 0.0], dtype=jnp.float64)
    return float(race_time_seconds(p, mdl))


def scenarios(final: dict) -> list:
    wheels = PARTS_N["wheelF"] + PARTS_N["wheelR"]
    rest = final["D20"] - wheels
    f = dict(final)
    return [
        ("session start: v2 parts (0.419 N, cad wheels 137.7)",
         dict(f, D20=0.419, L=0.21, I=137.7e-9)),
        ("final car, measured", f),
        ("+ film-covered carbon wheels (I 101.5)", dict(f, I=101.5e-9)),
        ("+ open carbon wheels, same drag (I 85.6)", dict(f, I=85.6e-9)),
        ("everything but wheels at half its drag", dict(f, I=85.6e-9, D20=wheels + rest / 2)),
        ("LIMIT: only the wheels make drag", dict(f, I=85.6e-9, D20=wheels)),
        ("LIMIT: + wheel drag halved (not achievable in the rules)",
         dict(f, I=85.6e-9, D20=wheels / 2)),
        ("LIMIT: no drag, no lift", dict(f, I=85.6e-9, D20=0.0, L=0.0)),
        ("LIMIT: no drag, no rolling friction, massless wheels",
         dict(f, I=0.0, D20=0.0, L=0.0, mu=0.0)),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", help="a run_car summary.json to take the final car from")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    from pipeline_interface import _add_sibling_packages_to_path  # noqa: F401
    from race_objective import build_smooth_sheet_model
    csv = a.csv or str(paths.PARTS["part2-simulation"] / "co2_thrust_data.csv")
    model = build_smooth_sheet_model(csv)
    final = dict(FINAL)
    if a.summary:
        S = json.loads(open(a.summary).read())
        c = S.get("cfd_final") or S["cfd_initial"]
        final.update(D20=c["D20_N"], L=c["L_N"], I=S["wheels"]["mean_I_kg_m2"])
    rows = []
    for name, p in scenarios(final):
        t1 = race_time(model, p["D20"], p["L"], p["I"], p["mu"], p["h"])
        t15 = race_time(model, p["D20"], p["L"], p["I"], p["mu"], p["h"], thrust=1.5)
        rows.append((name, p["D20"], t1, t15))
    base1, base15 = rows[1][2], rows[1][3]
    out = ["| car | D20 N | T model s | vs final ms | T real-thrust s | vs final ms |",
           "|---|---|---|---|---|---|"]
    for name, d, t1, t15 in rows:
        out.append(f"| {name} | {d:.3f} | {t1:.4f} | {1e3 * (t1 - base1):+.1f} | "
                   f"{t15:.4f} | {1e3 * (t15 - base15):+.1f} |")
    print("\n".join(out))
    return rows


if __name__ == "__main__":
    main()
