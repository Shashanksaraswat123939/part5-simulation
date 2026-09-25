"""
merge_results.py -- rank candidate records from one or more run folders.

    python merge_results.py RUN_DIR [RUN_DIR ...]

Rules (the contract Part 2's tests pin):
  * records describing DIFFERENT CARS (different W or x_front) are refused:
    merging shards that each re-ran Stage 1 compares different cars while
    looking reasonable. Fix: run every shard from one Stage-1 result
    (--stage1-in).
  * ranking is by T_penalized (the objective the optimiser minimised);
  * a car under the T3.6 48.0 g competition floor is EXCLUDED, and says so;
  * unscored candidates (CFD_failed, ...) are reported, never silently dropped;
  * every answer carries the noise caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import paths  # noqa: F401

CARTRIDGE_KG = 0.023
T36_FLOOR_KG = 0.048
NOISE_MS = 15


def _load(dirs):
    from candidate_record import read_candidate_record
    recs = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                recs.append(read_candidate_record(str(p)))
            except Exception:  # noqa: BLE001 -- not a record (assembly.json, ...)
                continue
    return recs


def _mass_g(rec):
    m = getattr(rec, "mass_report", None)
    return None if m is None else m.total_mass_kg * 1e3


def main(argv) -> int:
    if not argv:
        print("usage: merge_results.py RUN_DIR [RUN_DIR ...]")
        return 2
    recs = _load(argv)
    if not recs:
        print("no candidate records found")
        return 1
    cars = {(round(r.W_mm, 3), round(r.x_front_mm, 3)) for r in recs}
    if len(cars) > 1:
        print(f"REFUSED: these records describe DIFFERENT CARS {sorted(cars)}. "
              f"Run every shard from one Stage-1 result (--stage1-in) so they share "
              f"W and x_front.")
        return 1

    scored, unscored, excluded = [], [], []
    for r in recs:
        if r.T_penalized is None:
            unscored.append(r)
            continue
        m = _mass_g(r)
        if m is not None and (m / 1e3 - CARTRIDGE_KG) < T36_FLOOR_KG - 1e-9:
            excluded.append((r, m / 1e3 - CARTRIDGE_KG))
            continue
        scored.append(r)
    scored.sort(key=lambda r: (r.T_penalized, r.candidate_id))

    print(f"{'rank':>4} {'candidate':<28} {'d_halo':>7} {'T_pen s':>9} {'T_raw s':>9} "
          f"{'D20 N':>9} {'mass g':>8}")
    for i, r in enumerate(scored, 1):
        cfd = getattr(r, "cfd_force_report", None)
        d20 = f"{cfd.D20:.5f}" if cfd is not None else "-"
        m = _mass_g(r)
        print(f"{i:>4} {r.candidate_id:<28} {r.d_halo_mm:>7.2f} {r.T_penalized:>9.4f} "
              f"{(r.T_raw if r.T_raw is not None else float('nan')):>9.4f} {d20:>9} "
              f"{(f'{m:.2f}' if m is not None else '-'):>8}")
    for r, comp in excluded:
        print(f"EXCLUDED {r.candidate_id} d_halo={r.d_halo_mm:.2f}: underweight, "
              f"{comp*1000:.2f} g competition mass < {T36_FLOOR_KG*1000:.0f} g (T3.6)")
    for r in unscored:
        print(f"UNSCORED {r.candidate_id} d_halo={r.d_halo_mm:.2f}: {r.lifecycle_state}"
              f"{' -- ' + r.failure_reason if r.failure_reason else ''}")
    if scored:
        b = scored[0]
        print(f"BEST {b.candidate_id} d_halo={b.d_halo_mm:.2f} W={b.W_mm:.2f} "
              f"T_pen={b.T_penalized:.4f} s")
    print(f"NOTE: race-time differences under ~{NOISE_MS} ms are within the drag noise "
          f"of a single steady RANS solve; treat them as ties.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
