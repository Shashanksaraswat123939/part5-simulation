"""
pattern.py -- pattern search over the hybrid body, steered by direct CFD.

Team decision 2026-09-28. The adjoint could not steer (gradient check: right
sign 3/8, r = 0.45; frozen turbulence around the wheel wakes), but direct CFD of
a 2 mm sculpt mode moved D20 by up to 2.1 % against 0.08 % noise. So each
change is measured for real.

A round has two batches, each one GitHub Actions run (.github/workflows/pattern.yml):

  screen   the current car twice (the noise), and every parameter of one group
           at +step and -step. The skin is held at the current car's value so a
           job is one build + one CFD; cars are compared by race time at exactly
           the 48.2 g target (run_car's T_at_target_s), not at the few-tenths-
           of-a-gram they happen to weigh. Groups alternate: the 32 sculpt modes,
           then the loft (8 stations x b/zt/zb, sidepods, squareness, blend).
  confirm  candidates re-sized to 48.2 g (full mass loop): all winners together,
           the better half, the top 4, the current car again, and the current car
           with the step halved on its top 4. The best confirmed car becomes the
           current car only if it beats it by more than twice the noise.

A winner must beat the current car by more than twice the screen noise. If a
confirm finds nothing, the step halves. Stop when the step has halved twice
without a gain, or after MAX_ROUNDS.

    python pattern.py init                    # batch 0: the start car, sized, twice
    python pattern.py eval --id 5 --out o/    # one car (CI)
    python pattern.py advance results/        # read a batch's results, plan the next
    python pattern.py status
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths  # noqa: F401

HERE = Path(__file__).resolve().parent
STATE = HERE / "search" / "pattern_state.json"
BATCH = HERE / "search" / "pattern_batch.json"
MAX_ROUNDS = 6
# Floor on the noise estimate. Two base runs can agree to 0.02 ms, but three
# runs of one identical car in the round 2 confirm spread 1.2 ms (2026-09-30):
# solver run-to-run variation. A two-run estimate is too optimistic, so the
# floor is 0.5 ms and a winner must beat the base by more than 1 ms.
MIN_NOISE_S = 5e-4
LOFT = ([f"st_b{i}" for i in range(8)] + [f"st_zt{i}" for i in range(8)]
        + [f"st_zb{i}" for i in range(8)]
        + ["p", "s_b", "s_zt", "s_zb", "s_x0", "s_x1", "s_taper", "s_p", "blend_mm"])
# The front-wheel program (2026-09-30): the printed parts that shape the flow
# onto the front wheels, which carry ~44 % of the drag. name: (step, lo, hi).
# Bounds are the searchable range, not the rules: every candidate is built
# and audited, and one failing any gate is dropped like any other bad car.
PARTS = {
    "fw.aoa_deg": (2.0, -2.0, 14.0), "fw.z_chord_mm": (1.0, 6.5, 14.0),
    "fw.gap_to_wheel_mm": (1.0, 5.0, 10.0), "fw.half_span_mm": (2.0, 34.0, 42.5),
    "fw.flap_chord_mm": (1.0, 8.0, 11.0), "fw.flap_aoa_deg": (5.0, 0.0, 45.0),
    "fw.camber": (0.02, 0.0, 0.08),
    # the support (part4 beam_support.py): the plate's chord and least
    # thickness, the pod's length, roof and wall, the strip, the discs
    # (thicknesses start at the printer's 0.7 mm minimum; the generator
    # raises them itself where the loads need more)
    "sup.beam_w_mm": (2.0, 8.0, 24.0), "sup.beam_h_mm": (0.3, 0.7, 4.0),
    "sup.pod_len_mm": (2.0, 10.0, 20.0), "sup.pod_arch_mm": (1.0, 14.0, 19.0),
    "sup.pod_wall_mm": (0.2, 0.7, 1.5), "sup.strip": (1.0, 0.0, 1.0),
    "sup.disc_front": (1.0, 0.0, 1.0),
    "sup.disc_rear": (1.0, 0.0, 1.0), "sup.disc_r_mm": (1.0, 8.0, 12.0),
    # the disc inside the rim (+) or standing inboard of the wheel (-), and
    # the hubcap closing the wheel's outer side
    "sup.disc_recess_mm": (1.0, -2.5, 1.5), "sup.hubcap": (1.0, 0.0, 1.0),
    "nose.length_mm": (5.0, 10.0, 40.0), "nose.k": (0.15, 0.3, 1.0),
    "nose.p": (0.5, 2.0, 4.0), "nose.tip_z_mm": (2.0, 4.0, 20.0),
}
PART_FLAGS = {"fw": "--fwing-json", "sup": "--support-json", "nose": "--nose-json"}


def _pb():
    import param_body as pb
    return pb


def step0(name: str) -> float:
    if name in PARTS:
        return PARTS[name][0]
    if name.startswith("mode"):
        return 2.0
    if name in ("p", "s_p"):
        return 0.3
    if name in ("s_x0", "s_x1"):
        return 5.0
    if name == "blend_mm":
        return 1.0
    return 2.0


def bounds(name: str) -> tuple:
    pb = _pb()
    if name in PARTS:
        return PARTS[name][1:]
    if name.startswith("mode"):
        return (-6.0, 6.0)
    if name.startswith("st_b"):
        return (6.0, 20.0)
    if name.startswith("st_zt"):
        return (14.0, 48.0)
    if name.startswith("st_zb"):
        return (3.5, 23.0)
    if name == "blend_mm":
        return (0.0, 6.0)
    return pb.BOUNDS[name]


def group_name(state: dict) -> str:
    groups = state.get("groups", ["modes", "loft"])
    return groups[(state["round"] - state.get("focus_round", 0)) % len(groups)]


def group(state: dict) -> list:
    return {"modes": [f"mode{i}" for i in range(_pb().N_MODES)], "loft": LOFT,
            "parts": list(PARTS)}[group_name(state)]


def _part_default(name: str) -> float:
    import beam_support as bsm
    import nose as ns
    import wings as wg
    part, key = name.split(".")
    return float(getattr({"fw": wg.FrontWing(), "sup": bsm.BeamSupport(),
                          "nose": ns.NoseCone()}[part], key))


def get(d: dict, name: str) -> float:
    """d = {"params": body dict, "parts": {"fw": {...}, "sup": {...}, "nose": {...}}}."""
    if name in PARTS:
        part, key = name.split(".")
        return float(d["parts"].get(part, {}).get(key, _part_default(name)))
    bp = _pb().BodyParams.from_dict(d["params"])
    seq, i = _indexed(name)
    return float(getattr(bp, seq)[i]) if seq else float(getattr(bp, name))


def put(d: dict, name: str, v: float) -> dict:
    if name in PARTS:
        part, key = name.split(".")
        parts = {k: dict(x) for k, x in d["parts"].items()}
        parts.setdefault(part, {})[key] = (bool(round(v)) if key in ("disc_front", "disc_rear", "hubcap",
                                                                      "strip") else float(v))
        return {"params": d["params"], "parts": parts}
    from dataclasses import replace
    bp = _pb().BodyParams.from_dict(d["params"])
    seq, i = _indexed(name)
    if seq:
        vals = list(getattr(bp, seq))
        vals[i] = float(v)
        bp = replace(bp, **{seq: tuple(vals)})
    else:
        bp = replace(bp, **{name: float(v)})
    return {"params": bp.as_dict(), "parts": d["parts"]}


def _indexed(name: str):
    """("st_b", 3) for "st_b3", ("modes", 7) for "mode7", (None, None) for a scalar."""
    for key, attr in (("st_b", "st_b"), ("st_zt", "st_zt"), ("st_zb", "st_zb"), ("mode", "modes")):
        if name.startswith(key) and name[len(key):].isdigit():
            return attr, int(name[len(key):])
    return None, None


def _best(state: dict) -> dict:
    return {"params": state["best"], "parts": state.get("parts", {})}


def _case(i: int, tag: str, d: dict, sized: bool) -> dict:
    return {"id": i, "tag": tag, "params": d["params"], "parts": d["parts"], "sized": sized}


def _write_batch(state: dict, cases: list, phase: str):
    BATCH.write_text(json.dumps({"round": state["round"], "phase": phase,
                                 "skin_mm": state.get("skin_mm"), "cases": cases}, indent=1))
    (HERE / "search" / "current.json").write_text(json.dumps(
        {"round": state["round"], "phase": phase, "ids": [c["id"] for c in cases]}))


def init():
    pb = _pb()
    (HERE / "search").mkdir(exist_ok=True)
    start = pb.BodyParams()
    state = {"round": 0, "phase": "confirm", "best": start.as_dict(), "best_T": None,
             "skin_mm": None, "scale": 1.0, "fails": 0, "history": []}
    STATE.write_text(json.dumps(state, indent=1))
    cases = [{"id": 0, "tag": "start", "params": start.as_dict(), "sized": True},
             {"id": 1, "tag": "start_repeat", "params": start.as_dict(), "sized": True}]
    _write_batch(state, cases, "confirm")
    return state


def evaluate(case_id: int, out: Path) -> dict:
    import run_car as rc
    B = json.loads(BATCH.read_text())
    c = next(c for c in B["cases"] if c["id"] == case_id)
    # beam: the parametrised team architecture; the v2 CAD rear support fails
    # T7.13 (its disc blocks the hang-test claw), so it cannot be the base.
    args = ["--out", str(out), "--cfd", "--res", "medium", "--support", "beam",
            "--body-json", json.dumps(c["params"])]
    for part, flag in PART_FLAGS.items():
        if c.get("parts", {}).get(part):
            args += [flag, json.dumps(c["parts"][part])]
    if not c["sized"]:
        args += ["--skin-mm", str(B["skin_mm"])]
    row = {"id": case_id, "tag": c["tag"], "round": B["round"], "phase": B["phase"],
           "params": c["params"], "parts": c.get("parts", {}), "sized": c["sized"], "ok": False}
    try:
        S = rc.main(args)
        ci = S["cfd_initial"]
        # screening cars are not re-sized, and a torn export file does not
        # change the flow: both only gate the sized cars a confirm can accept
        failed = [k for k in S["legality"]["summary"]["failed"]
                  if c["sized"] or k not in ("T3.6_mass", "manufacture_files_are_solids")]
        row.update(T_s=ci["T_at_target_s"], T_raw_s=ci["T_raw_s"], D20_N=ci["D20_N"],
                   converged=ci["converged"],
                   mass_g=S["mass"].get("manufactured_mass_g", S["mass"]["competition_mass_g"]),
                   skin_mm=S["body"].get("build", {}).get("skin_offset_mm"),
                   failed=failed, ok=bool(ci["converged"] and not failed))
    except (Exception, SystemExit) as exc:  # noqa: BLE001 -- a failed car is a result
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pattern_r{B['round']}_{B['phase']}_{case_id}.json").write_text(
        json.dumps(row, indent=1, default=str))
    return row


def _results(root: Path, B: dict) -> list:
    rows = [json.loads(p.read_text()) for p in Path(root).rglob("pattern_*.json")]
    return [r for r in rows if r["round"] == B["round"] and r["phase"] == B["phase"]]


def average_repeats(rows: list) -> list:
    """Cars with identical parameters are repeat measurements of one car:
    return one row each, at their mean time. Taking the best of several
    identical cars rewards solver noise (round 2 confirm, 2026-09-30)."""
    groups = {}
    for r in rows:
        key = json.dumps([r["params"], r.get("parts", {})], sort_keys=True)
        groups.setdefault(key, []).append(r)
    return [dict(g[0], T_s=float(np.mean([r["T_s"] for r in g])),
                 tag="+".join(r["tag"] for r in g), n_repeats=len(g))
            for g in groups.values()]


def advance(root: Path) -> str:
    state = json.loads(STATE.read_text())
    B = json.loads(BATCH.read_text())
    rows = _results(root, B)
    by = {r["tag"]: r for r in rows}
    msg = []
    if B["phase"] == "screen":
        bases = [by[t]["T_s"] for t in ("base", "base_repeat") if t in by and by[t]["ok"]]
        if not bases:
            raise SystemExit("no valid base car in the screen batch")
        T0 = float(np.mean(bases))
        noise = max(abs(bases[0] - bases[-1]), MIN_NOISE_S)
        best = _best(state)
        wins = []
        for n in group(state):
            opts = [(by[f"{n}{s}"]["T_s"], d) for s, d in (("+", 1), ("-", -1))
                    if f"{n}{s}" in by and by[f"{n}{s}"]["ok"]]
            if opts:
                t, d = min(opts)
                if t < T0 - 2 * noise:
                    wins.append((t - T0, n, d))
        wins.sort()
        msg.append(f"screen r{state['round']}: base {T0:.5f} s, noise {noise * 1e3:.3f} ms, "
                   f"{len(wins)} winners: " + ", ".join(f"{n}{'+' if d > 0 else '-'} "
                                                          f"{dt * 1e3:+.2f} ms" for dt, n, d in wins[:10]))

        def moved(sel, frac=1.0):
            p = best
            for _dt, n, d in sel:
                lo, hi = bounds(n)
                v = get(p, n) + d * frac * state["scale"] * step0(n)
                p = put(p, n, float(np.clip(v, lo, hi)))
            return p

        cases = [_case(0, "current", best, True)]
        if wins:
            cases += [_case(1, "all_winners", moved(wins), True),
                      _case(2, "top_half", moved(wins[:max(1, len(wins) // 2)]), True),
                      _case(3, "top4", moved(wins[:4]), True),
                      _case(4, "top4_half_step", moved(wins[:4], 0.5), True)]
        state["phase"] = "confirm"
        state["screen_noise_s"] = noise
        state["history"].append({"round": state["round"], "screen_T0": T0, "noise": noise,
                                 "winners": [(n, d, dt) for dt, n, d in wins]})
        STATE.write_text(json.dumps(state, indent=1))
        _write_batch(state, cases, "confirm")
        return "\n".join(msg)

    # confirm
    good = average_repeats([r for r in rows if r["ok"]])
    if not good:
        raise SystemExit("no valid car in the confirm batch")
    cur = [r["T_s"] for r in good if r["tag"] in ("current", "start", "start_repeat")]
    top = min(good, key=lambda r: r["T_s"])
    noise = max(state.get("screen_noise_s", MIN_NOISE_S), MIN_NOISE_S)
    # Compare within this batch: the current car was re-run beside the candidates.
    ref = float(np.mean(cur)) if cur else state["best_T"]
    if state["best_T"] is None:                            # first confirm: just the start car
        state["best_T"] = ref
        state["skin_mm"] = float(np.mean([r["skin_mm"] for r in good if r["skin_mm"] is not None]))
        msg.append(f"start car: {ref:.5f} s at 48.2 g, skin {state['skin_mm']:.2f} mm")
    elif top["T_s"] < ref - 2 * noise and top["tag"] not in ("current",):
        state["best"], state["best_T"] = top["params"], top["T_s"]
        state["parts"] = top.get("parts", {})
        state["skin_mm"] = top["skin_mm"]
        state["fails"] = 0
        msg.append(f"accepted {top['tag']}: {top['T_s']:.5f} s ({(top['T_s'] - ref) * 1e3:+.2f} ms)")
    else:
        state["scale"] *= 0.5
        state["fails"] += 1
        msg.append(f"no gain (best {top['tag']} {top['T_s']:.5f} vs {ref:.5f}); step now "
                   f"x{state['scale']}")
    state["round"] += 1 if state["history"] else 0
    stop = (state["fails"] >= 2
            or state["round"] - state.get("focus_round", 0) >= MAX_ROUNDS)
    STATE.write_text(json.dumps(state, indent=1, default=float))
    if stop:
        msg.append(f"STOP: best {state['best_T']:.5f} s")
        return "\n".join(msg)
    msg.append(plan_screen(state))
    return "\n".join(msg)


def plan_screen(state: dict) -> str:
    """The current car twice, and every parameter of this round's group at
    +/- one step."""
    best = _best(state)
    cases = [_case(0, "base", best, False), _case(1, "base_repeat", best, False)]
    for n in group(state):
        lo, hi = bounds(n)
        for s, d in (("+", 1), ("-", -1)):
            v = float(np.clip(get(best, n) + d * state["scale"] * step0(n), lo, hi))
            if v != get(best, n):
                cases.append(_case(len(cases), f"{n}{s}", put(best, n, v), False))
    state["phase"] = "screen"
    STATE.write_text(json.dumps(state, indent=1, default=float))
    _write_batch(state, cases, "screen")
    return (f"next: screen r{state['round']} ({len(cases)} cars, group {group_name(state)}, "
            f"step x{state['scale']})")


def focus(groups: list) -> str:
    """Point the search at other parameter groups (["parts"] for the
    front-wheel program): full step, fresh fail count, the round budget
    restarts, and a screen is planned around the current car. Any pending
    confirm must be advanced first."""
    state = json.loads(STATE.read_text())
    if json.loads(BATCH.read_text())["phase"] == "confirm" and state.get("history"):
        raise SystemExit("a confirm batch is pending: advance it first")
    state.update(groups=groups, scale=1.0, fails=0, focus_round=state["round"])
    return plan_screen(state)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init")
    e = sp.add_parser("eval")
    e.add_argument("--id", type=int, required=True)
    e.add_argument("--out", required=True)
    a_ = sp.add_parser("advance")
    a_.add_argument("root")
    f_ = sp.add_parser("focus")
    f_.add_argument("groups", nargs="+", choices=("modes", "loft", "parts"))
    sp.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        init()
        print("batch 0: the start car, sized, twice")
    elif a.cmd == "eval":
        print(json.dumps(evaluate(a.id, Path(a.out)), default=str)[:1500])
    elif a.cmd == "advance":
        print(advance(Path(a.root)))
    elif a.cmd == "focus":
        print(focus(a.groups))
    else:
        s = json.loads(STATE.read_text())
        print(json.dumps({k: v for k, v in s.items() if k not in ("best", "history")}, default=str))


if __name__ == "__main__":
    main()
