"""
body_search.py -- find the fastest parametric body with real CFD, in batches.

    python body_search.py plan --round 0 [--n 16]          # -> search/round_0.json
    python body_search.py eval --round 0 --id 3 --out o/    # one car, CFD + race time
    python body_search.py collect results/                 # ranked table of every round

Each round is one GitHub Actions run (.github/workflows/body_search.yml): every
candidate is a job, a whole car through run_car (Part 1 param_body -> Part 4
parts -> legality -> medium-mesh CFD with every part -> race objective).

Round 0 is a Latin hypercube over param_body.BOUNDS plus the default body.
Later rounds are a (mu, lambda) evolution: the best legal cars so far are the
parents, children are Gaussian steps (sigma shrinking by round) and uniform
crossovers of the two best. The best car so far is re-run each round, so its
number carries a repeat. Race time is the objective, not drag: mass, ballast
and COM come with every car.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import paths  # noqa: F401

HERE = Path(__file__).resolve().parent
SEARCH = HERE / "search"
SIGMA0, DECAY, PARENTS = 0.15, 0.7, 4


def _pb():
    import param_body as pb
    return pb


def load_results(root: Path) -> list:
    rows = []
    for p in sorted(Path(root).rglob("bodysearch_*.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def _legal(r) -> bool:
    # Converged too: round 0's 0.10 "won" with -0.16 N of body drag from a
    # solve drifting 31 % -- an oscillating flow, not a fast car.
    return (r.get("legal", False) and r.get("T_raw_s") is not None
            and r.get("converged", False))


def plan(rnd: int, n: int, results: Path | None) -> list:
    pb = _pb()
    rng = np.random.default_rng(1000 + rnd)
    keys = list(pb.BOUNDS)
    lo = np.array([pb.BOUNDS[k][0] for k in keys])
    hi = np.array([pb.BOUNDS[k][1] for k in keys])
    if rnd == 0:
        cands = [pb.BodyParams().as_dict()] + pb.sample(rng, n - 1)
    else:
        done = [r for r in load_results(results) if _legal(r)] if results else []
        if not done:
            raise SystemExit("no legal results to evolve from")
        done.sort(key=lambda r: r["T_raw_s"])
        par = [np.array([r["params"][k] for k in keys]) for r in done[:PARENTS]]
        sigma = SIGMA0 * DECAY ** (rnd - 1) * (hi - lo)
        cands = [done[0]["params"]]                         # the best, re-run
        w = np.array([PARENTS - i for i in range(len(par))], float)
        w /= w.sum()
        while len(cands) < n:
            if len(cands) % 5 == 4 and len(par) > 1:        # crossover of the two best
                mask = rng.random(len(keys)) < 0.5
                x = np.where(mask, par[0], par[1])
            else:
                x = par[rng.choice(len(par), p=w)] + rng.normal(0, 1, len(keys)) * sigma
            x = np.clip(x, lo, hi)
            cands.append({k: float(v) for k, v in zip(keys, x)})
    out = [{"id": i, "round": rnd, "params": c} for i, c in enumerate(cands)]
    SEARCH.mkdir(exist_ok=True)
    (SEARCH / f"round_{rnd}.json").write_text(json.dumps(out, indent=1))
    (SEARCH / "current.json").write_text(json.dumps({"round": rnd, "ids": [c["id"] for c in out]}))
    return out


def evaluate(rnd: int, cid: int, out: Path, res: str) -> dict:
    import run_car as rc
    cand = next(c for c in json.loads((SEARCH / f"round_{rnd}.json").read_text())
                if c["id"] == cid)
    row = {"round": rnd, "id": cid, "params": cand["params"], "legal": False, "T_raw_s": None}
    try:
        S = rc.main(["--out", str(out), "--cfd", "--res", res,
                     "--body-json", json.dumps(cand["params"])])
        c = S["cfd_initial"]
        row.update(T_raw_s=c["T_raw_s"], D20_N=c["D20_N"], L_N=c["L_N"],
                   converged=c["converged"], parts={k: v["D_N"] for k, v in c["parts"].items()},
                   mass=S["mass"], legality=S["legality"]["summary"],
                   legal=S["legality"]["summary"]["n_failed"] == 0)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 -- a failed car is a result
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bodysearch_r{rnd}_{cid}.json").write_text(json.dumps(row, indent=1, default=str))
    return row


def collect(root: Path) -> str:
    rows = load_results(root)
    rows.sort(key=lambda r: (not _legal(r), r["T_raw_s"] if r["T_raw_s"] is not None else 9))
    lines = ["| round.id | legal | T s | D20 N | body N | wheels N | ballast g | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r.get("parts", {})
        note = r.get("error", "") or ", ".join(r.get("legality", {}).get("failed", []))
        lines.append(
            f"| {r['round']}.{r['id']} | {'yes' if _legal(r) else 'NO'} | "
            f"{r['T_raw_s'] if r['T_raw_s'] is None else round(r['T_raw_s'], 4)} | "
            f"{round(r.get('D20_N', 0), 4)} | {round(p.get('car', 0), 4)} | "
            f"{round(p.get('wheelF', 0) + p.get('wheelR', 0), 4)} | "
            f"{round(r.get('mass', {}).get('ballast_g', 0), 2)} | {note[:80]} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--results", default=None)
    e = sp.add_parser("eval")
    e.add_argument("--round", type=int, required=True)
    e.add_argument("--id", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--res", default="medium")
    c = sp.add_parser("collect")
    c.add_argument("root")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cands = plan(a.round, a.n, Path(a.results) if a.results else None)
        print(f"round {a.round}: {len(cands)} candidates")
    elif a.cmd == "eval":
        print(json.dumps(evaluate(a.round, a.id, Path(a.out), a.res), default=str)[:2000])
    else:
        print(collect(Path(a.root)))


if __name__ == "__main__":
    main()
