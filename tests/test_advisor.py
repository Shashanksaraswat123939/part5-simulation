"""Part 5 checks: legality engine, recommendations, and a no-CFD run of the whole chain."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402,F401


def _box_car(td):
    import trimesh
    b = trimesh.creation.box(extents=[0.19, 0.056, 0.042])
    b.apply_translation([0.030 + 0.095, 0.0, 0.0025 + 0.021])
    half = trimesh.intersections.slice_mesh_plane(b, [0, 1, 0], [0, 0, 0], cap=True)
    half.export(f"{td}/body.stl", file_type="stl_ascii")
    return f"{td}/body.stl"


def test_legality_measures_width_height_overhang_and_block():
    import assembly
    import legality
    with tempfile.TemporaryDirectory() as td:
        body = _box_car(td)
        a = assembly.build(120.3, 46.0, 43.72, body, f"{td}/asm")
        r = legality.check(body, a, {"competition_mass_g": 48.2, "ballast_g": 3.0,
                                     "capacity_g": 15.8}, field_bodies=1)
        assert r["T3.4_width_min"]["pass"] and r["T3.5_height"]["pass"]
        assert abs(float(r["block_length"]["note"].split()[0]) - 190.0) < 0.5
        assert r["block_width"]["pass"] and r["T4.1_single_body"]["pass"]
        assert r["T3.6_mass"]["margin"] > 0
        # The v2 rear-support CAD bottoms at 1.40 mm; Part 4 trims it to 1.51.
        assert r["T3.7_clearance"]["pass"]
        r2 = legality.check(body, a, {"competition_mass_g": 47.8, "ballast_g": 15.8,
                                      "capacity_g": 15.8}, field_bodies=2)
        assert not r2["T3.6_mass"]["pass"] and not r2["T4.1_single_body"]["pass"]


def test_t55_probe_sees_a_missing_wall():
    import math
    import numpy as np
    import trimesh
    import legality
    solid = trimesh.creation.box(extents=[0.06, 0.04, 0.04])
    solid.apply_translation([0.19, 0.0, 0.035])
    half = trimesh.intersections.slice_mesh_plane(solid, [0, 1, 0], [0, 0, 0], cap=True)
    assert legality.t55_wall_fraction(half, rear_face_mm=220.0) > 0.999
    thin = trimesh.creation.box(extents=[0.06, 0.021, 0.04])      # 10.5 mm half-width
    thin.apply_translation([0.19, 0.0, 0.035])
    half2 = trimesh.intersections.slice_mesh_plane(thin, [0, 1, 0], [0, 0, 0], cap=True)
    assert legality.t55_wall_fraction(half2, rear_face_mm=220.0) < 0.95


def test_recommendations_follow_the_numbers():
    import run_car
    legal = {"T3.7_clearance": {"pass": False, "margin": -0.1, "note": ""}}
    mstate = {"regime": "absorbing", "ballast_g": 4.0, "capacity_g": 15.8}
    cfd = {"D20_N": 0.40, "converged": True, "gradients": {"dT_dD20": 0.45},
           "parts": {"wheelF": {"D_N": 0.14}, "wheelR": {"D_N": 0.14}, "car": {"D_N": 0.07},
                     "fwing": {"D_N": 0.05}}}
    rec = run_car.recommend(legal, mstate, cfd, {"wheels: carbon_rim": -5.0, "drag -10 %": -12})
    txt = " ".join(rec)
    assert rec[0].startswith("FIX LEGALITY FIRST") and "T3.7" in rec[0]
    assert "Wheels carry 70 %" in txt and "carbon_rim" in txt and "Ballast absorbs" in txt


def test_whole_chain_without_cfd():
    import run_car
    with tempfile.TemporaryDirectory() as td:
        S = run_car.main(["--out", td, "--stage1-iters", "20", "--stage1-mm", "3.0",
                          "--cfd-mm", "2.0"])
        assert (Path(td) / "report.md").exists() and (Path(td) / "summary.json").exists()
        assert S["legality"]["summary"]["n_checks"] > 40
        assert S["mass"]["regime"] == "none"          # no ballast (team spec 2026-09-27)
        assert S["mass"]["ballast_g"] == 0.0
        assert len(json.loads((Path(td) / "parts" / "assembly.json").read_text())
                   ["extra_surfaces"]) == 7


def test_pattern_confirm_averages_identical_cars():
    import pattern
    p1, p2 = {"modes": [1.0]}, {"modes": [2.0]}
    rows = pattern.average_repeats([
        {"params": p1, "T_s": 1.0, "tag": "a"}, {"params": p1, "T_s": 2.0, "tag": "b"},
        {"params": p2, "T_s": 1.2, "tag": "c"}])
    by = {r["tag"]: r for r in rows}
    assert by["a+b"]["T_s"] == 1.5 and by["a+b"]["n_repeats"] == 2
    assert by["c"]["T_s"] == 1.2 and len(rows) == 2


def test_pattern_parts_screen_carries_part_changes_to_the_confirm():
    import pattern
    import param_body as pb
    with tempfile.TemporaryDirectory() as td:
        here = Path(td)
        (here / "search").mkdir()
        old = pattern.HERE, pattern.STATE, pattern.BATCH
        pattern.HERE, pattern.STATE, pattern.BATCH = (
            here, here / "search" / "s.json", here / "search" / "b.json")
        try:
            body = pb.BodyParams().to_hybrid().as_dict()
            pattern.STATE.write_text(json.dumps({"round": 4, "phase": "screen", "best": body,
                                                 "best_T": 1.57, "skin_mm": 1.0, "scale": 1.0,
                                                 "fails": 0, "history": [{"round": 3}]}))
            pattern.BATCH.write_text(json.dumps({"round": 3, "phase": "screen", "cases": []}))
            pattern.focus(["parts"])
            B = json.loads(pattern.BATCH.read_text())
            tags = {c["tag"]: c for c in B["cases"]}
            assert B["phase"] == "screen" and "fw.aoa_deg+" in tags and "nose.length_mm-" in tags
            assert "sup.beam_d_mm-" not in tags                  # 7 mm is the lower bound
            assert tags["sup.beam_d_mm+"]["parts"]["sup"] == {"beam_w_mm": 8.0, "beam_h_mm": 8.0}
            assert tags["sup.disc_front+"]["parts"]["sup"]["disc_front"] is True
            assert tags["fw.aoa_deg+"]["parts"]["fw"]["aoa_deg"] == 8.0
            rows = [{"round": 4, "phase": "screen", "tag": c["tag"], "ok": True,
                     "params": c["params"], "parts": c["parts"],
                     "T_s": 1.570 - (0.005 if c["tag"] == "fw.aoa_deg+" else 0.0)}
                    for c in B["cases"]]
            (here / "res").mkdir()
            for i, r in enumerate(rows):
                (here / "res" / f"pattern_{i}.json").write_text(json.dumps(r))
            pattern.advance(here / "res")
            C = json.loads(pattern.BATCH.read_text())
            win = next(c for c in C["cases"] if c["tag"] == "all_winners")
            assert C["phase"] == "confirm" and win["parts"]["fw"]["aoa_deg"] == 8.0
        finally:
            pattern.HERE, pattern.STATE, pattern.BATCH = old


if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _fails = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)(); print("PASS", _n)
        except Exception as e:  # noqa: BLE001
            _fails += 1; print("FAIL", _n, "->", repr(e))
    print(f"{_fails} failed")
    sys.exit(1 if _fails else 0)
