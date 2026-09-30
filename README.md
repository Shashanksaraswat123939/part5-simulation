# Part 5 — Design advisor and pipeline

Runs one car through all five parts, audits it like a scrutineer, measures it in CFD
with every part in the flow, searches the body shape, and writes a report with ranked,
quantified recommendations.

The car as built today (team decisions 2026-09-27/28, see
`docs/BODY_GENERATION_DECISION.md`): a parametric "hybrid" body (Part 1
`param_body`: 8-station loft + 32 sculpt modes), milled with a 6.25 mm ball end
from four sides, printed PA12 positives glued into milled pockets (Part 4
`joints`), one-piece beam supports with printed stub axles and one 3x6x2.5 bearing
per wheel, the team's STL wheels in PA12, a PA12 nose shell, and NO ballast: the
body volume is sized so the car weighs 48.2 g.

| file | what |
|---|---|
| `run_car.py` | the pipeline: body (Part 1) → parts + joints (Part 4) → legality → CFD (Part 2) → optional adjoint loop (Part 3) → report |
| `legality.py` | 72 checks on the assembled car: T3.4–T3.7, T4.1, T5.5 (on the milled body), T7.3, T7.13 hang-test claw, T8.2, T8.5.1, T9.4.2, Model Block, every part attached, plus every Part 4 gate |
| `pattern.py` | the body optimiser: pattern search over the 32 modes and the loft, every step measured by medium-mesh CFD on GitHub (`.github/workflows/pattern.yml`, pushed to a `pattern/**` branch); state in `search/` |
| `hybrid.py` | parameter get/put for the hybrid body; the adjoint-steered variant that failed its gradient check (kept as the record) |
| `sweep.py` | wheel and wing variants on a fixed body, medium-mesh CFD, ranked by race time (`sweep.yml`) |
| `sign_check.py` | one aero-only step along +adjoint and −adjoint, three forward solves: does descent lower drag? |
| `merge_results.py` | ranks candidate records (T_penalized, underweight excluded, different cars refused) |
| `.github/workflows/ci.yml` | all five test suites + the start car with real OpenFOAM on GitHub Actions |

```bash
python run_car.py --out results/ --body-json '{}' --support beam                    # no CFD, ~5 min (mass sizing)
python run_car.py --out results/ --body-json '{}' --support beam --cfd --res medium # + forward CFD with all parts
python pattern.py init && git push origin HEAD:pattern/r0        # batch 0 on GitHub
python pattern.py advance results/                                # read a batch, plan the next
python run_car.py --out results/                                  # the older level-set carve, v2 CAD supports
```

The folders `part1-simulation` … `part4-simulation` must sit beside this one.
