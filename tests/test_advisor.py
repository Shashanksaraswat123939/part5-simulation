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
        assert S["mass"]["regime"] in ("absorbing", "heavy", "full")
        assert len(json.loads((Path(td) / "parts" / "assembly.json").read_text())
                   ["extra_surfaces"]) == 7


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
