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


def _pb():
    import param_body as pb
    return pb


def step0(name: str) -> float:
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


def group(round_no: int) -> list:
    return [f"mode{i}" for i in range(_pb().N_MODES)] if round_no % 2 == 0 else LOFT


def _get_put():
    import hybrid as hy
    return hy.get, hy.put


def _write_batch(state: dict, cases: list, phase: str):
    BATCH.write_text(json.dumps({"round": state["round"], "phase": phase,
                                 "skin_mm": state.get("skin_mm"), "cases": cases}, indent=1))
    (HERE / "search" / "current.json").write_text(json.dumps(
        {"round": state["round"], "phase": phase, "ids": [c["id"] for c in cases]}))


def init():
    pb = _pb()
    (HERE / "search").mkdir(exist_ok=True)
    start = pb.BodyParams().to_hybrid()
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
    if not c["sized"]:
        args += ["--skin-mm", str(B["skin_mm"])]
    row = {"id": case_id, "tag": c["tag"], "round": B["round"], "phase": B["phase"],
           "params": c["params"], "sized": c["sized"], "ok": False}
    try:
        S = rc.main(args)
        ci = S["cfd_initial"]
        failed = [k for k in S["legality"]["summary"]["failed"]
                  if not (k == "T3.6_mass" and not c["sized"])]   # screening cars are not re-sized
        row.update(T_s=ci["T_at_target_s"], T_raw_s=ci["T_raw_s"], D20_N=ci["D20_N"],
                   converged=ci["converged"], mass_g=S["mass"]["competition_mass_g"],
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
        groups.setdefault(json.dumps(r["params"], sort_keys=True), []).append(r)
    return [dict(g[0], T_s=float(np.mean([r["T_s"] for r in g])),
                 tag="+".join(r["tag"] for r in g), n_repeats=len(g))
            for g in groups.values()]


def advance(root: Path) -> str:
    get, put = _get_put()
    pb = _pb()
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
        best = pb.BodyParams.from_dict(state["best"])
        wins = []
        for n in group(state["round"]):
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
            return p.as_dict()

        cases = [{"id": 0, "tag": "current", "params": best.as_dict(), "sized": True}]
        if wins:
            cases += [{"id": 1, "tag": "all_winners", "params": moved(wins), "sized": True},
                      {"id": 2, "tag": "top_half", "params": moved(wins[:max(1, len(wins) // 2)]),
                       "sized": True},
                      {"id": 3, "tag": "top4", "params": moved(wins[:4]), "sized": True},
                      {"id": 4, "tag": "top4_half_step", "params": moved(wins[:4], 0.5),
                       "sized": True}]
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
        state["skin_mm"] = top["skin_mm"]
        state["fails"] = 0
        msg.append(f"accepted {top['tag']}: {top['T_s']:.5f} s ({(top['T_s'] - ref) * 1e3:+.2f} ms)")
    else:
        state["scale"] *= 0.5
        state["fails"] += 1
        msg.append(f"no gain (best {top['tag']} {top['T_s']:.5f} vs {ref:.5f}); step now "
                   f"x{state['scale']}")
    state["round"] += 1 if state["history"] else 0
    stop = state["fails"] >= 2 or state["round"] >= MAX_ROUNDS
    STATE.write_text(json.dumps(state, indent=1, default=float))
    if stop:
        msg.append(f"STOP: best {state['best_T']:.5f} s")
        return "\n".join(msg)
    # next screen
    best = pb.BodyParams.from_dict(state["best"])
    cases = [{"id": 0, "tag": "base", "params": best.as_dict(), "sized": False},
             {"id": 1, "tag": "base_repeat", "params": best.as_dict(), "sized": False}]
    for n in group(state["round"]):
        lo, hi = bounds(n)
        for s, d in (("+", 1), ("-", -1)):
            v = float(np.clip(get(best, n) + d * state["scale"] * step0(n), lo, hi))
            if v != get(best, n):
                cases.append({"id": len(cases), "tag": f"{n}{s}", "params": put(best, n, v).as_dict(),
                              "sized": False})
    state["phase"] = "screen"
    STATE.write_text(json.dumps(state, indent=1, default=float))
    _write_batch(state, cases, "screen")
    msg.append(f"next: screen r{state['round']} ({len(cases)} cars, group "
               f"{'modes' if state['round'] % 2 == 0 else 'loft'}, step x{state['scale']})")
    return "\n".join(msg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init")
    e = sp.add_parser("eval")
    e.add_argument("--id", type=int, required=True)
    e.add_argument("--out", required=True)
    a_ = sp.add_parser("advance")
    a_.add_argument("root")
    sp.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        init()
        print("batch 0: the start car, sized, twice")
    elif a.cmd == "eval":
        print(json.dumps(evaluate(a.id, Path(a.out)), default=str)[:1500])
    elif a.cmd == "advance":
        print(advance(Path(a.root)))
    else:
        s = json.loads(STATE.read_text())
        print(json.dumps({k: v for k, v in s.items() if k not in ("best", "history")}, default=str))


if __name__ == "__main__":
    main()
