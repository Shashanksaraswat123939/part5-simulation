# Part 5 — Design advisor and pipeline (simscale branch)

This branch runs every CFD solve on **SimScale** through the SimScale Python SDK;
there is no OpenFOAM and no adjoint here. The flow domain (wind-tunnel box minus
the half car, every part its own named face) is built on the runner, then SimScale
imports, meshes (boundary layers from a 15 µm first cell for y+ <= 2) and solves it.
Main keeps the OpenFOAM version.

Set once, in the repository settings:
- secrets `SIMSCALE_API_KEY` and `SIMSCALE_PROJECT_ID`
- optionally the variable `SIMSCALE_MAX_PARALLEL` (default 4 runs at once)

First live step: Actions > ci > Run workflow with `simscale = probe`. It imports the
start car's domain and prints how SimScale named its faces, with no mesh and no run.

The car as built today (team decisions 2026-09-27/28, see
`docs/BODY_GENERATION_DECISION.md`): a parametric "hybrid" body (Part 1
`param_body`: 8-station loft + 32 sculpt modes), milled with a 6.25 mm ball end
from four sides, printed PA12 positives glued into milled pockets (Part 4
`joints`), one-piece beam supports with printed stub axles and one 3x6x2.5 bearing
per wheel, the team's STL wheels in PA12, a PA12 nose shell, and NO ballast: the
body volume is sized so the car weighs 48.2 g.

| file | what |
|---|---|
| `run_car.py` | the pipeline: parametric body (Part 1) sized so the car as manufactured weighs 48.2 g → parts, pockets and manufacturing files (Part 4) → legality → SimScale CFD with every part (Part 2) → report |
| `legality.py` | 72 checks on the assembled car: T3.4–T3.7, T4.1, T5.5 (on the milled body), T7.3, T7.13 hang-test claw, T8.2, T8.5.1, T9.4.2, Model Block, T3.6 on the manufactured mass, every part attached, every manufacturing file a closed solid, plus every Part 4 gate |
| `pattern.py` | the optimiser: pattern search over the body (32 modes, the loft) and the printed parts (front wing, beam support, nose), every car solved on SimScale (`pattern.yml`, pushed to a `simscale-pattern/**` branch); state in `search/` |
| `merge_results.py` | ranks candidate records from the adjoint loop |
| `render.py` | a picture of a built car |
| `.github/workflows/ci.yml` | all five test suites and the manufacturing and flow-domain check on every push; by hand, the SimScale probe or the start car on SimScale |

```bash
python run_car.py --out results/                                  # the start car, no CFD, ~6 min (sizing)
python run_car.py --out results/ --body-json "$(cat checks/round2_car.json)" --cfd --res medium
python pattern.py advance results/                                # read a batch, plan the next
python pattern.py focus parts                                     # switch the search to the printed parts
python -m pytest tests -q
```

The folders `part1-simulation` … `part4-simulation` must sit beside this one.
