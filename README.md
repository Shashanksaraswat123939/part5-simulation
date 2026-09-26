# Part 5 — Design advisor and pipeline

Runs one car through all five parts, audits it like a scrutineer, measures it in CFD
with every part in the flow, optimises the body, and writes a report with ranked,
quantified recommendations.

| file | what |
|---|---|
| `run_car.py` | the pipeline: body (Part 1) → parts (Part 4) → legality → CFD (Part 2) → optimise (Part 3) → report |
| `legality.py` | ~55 checks on the assembled car: T3.4–T3.7, T4.1, T5.5, T7.3, T8.2, T8.5.1, T9.4.2, Model Block, T3.6 with ballast, every part attached to the body, plus every Part 4 gate |
| `sweep.py` | wheel and wing variants on a fixed body, medium-mesh CFD, ranked by race time (`.github/workflows/sweep.yml`) |
| `sign_check.py` | one aero-only step along +adjoint and −adjoint, three forward solves: does descent lower drag? |
| `merge_results.py` | ranks candidate records (T_penalized, underweight excluded, different cars refused) |
| `.github/workflows/ci.yml` | all five test suites + three real-OpenFOAM car runs on GitHub Actions |

```bash
python run_car.py --out results/                     # no CFD, ~20 s
python run_car.py --out results/ --cfd --res medium  # + forward CFD with all parts
python run_car.py --out results/ --cfd --optimise 4 --final-cfd --res coarse
```

The folders `part1-simulation` … `part4-simulation` must sit beside this one.
