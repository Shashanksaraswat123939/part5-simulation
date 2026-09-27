# How the car body gets its shape — methods, why not, and challenges

Status: **proposal, awaiting the team's decision** (2026-09-27).
Scope: the machined main body (Part 1). The printed parts (Part 4) are covered only
where they constrain the body.

---

## 1. What the body has to satisfy

### 1.1 Team manufacturing and design facts (given 2026-09-27)

| Item | Fact | Consequence for the body |
|---|---|---|
| Body | 3-axis milling, **6.25 mm ball-end** cutter | every inside corner ≥ 3.125 mm radius; no slot narrower than 6.25 mm |
| Setups | **top, bottom, left, right** (4) | every surface must be reachable along ±z or ±y; nothing that can only be reached from the front or back, bar the drilled cartridge bore |
| Printed parts | **SLS PA12**, positives glued **into** negatives in the body | the body carries pockets/channels; printed parts carry the matching plugs |
| Wheel supports | one piece per axle, left to right, stub axles printed in, **3×6×2.5 bearings** pressed onto the stubs | the body needs a full-width channel per axle. The discs and stubs at the ends make the part wider than the beam, so it drops in **from below**; it can't slide through a side slot |
| Nose cone | SLS PA12, **shell** (hollow with drain holes) | the body ends at Ref A; the nose is a separate part glued into a short pocket |
| Wheels | SLS PA12 | (open: the repo's `hardware_cad/*wheel.stl` or a parametric PA12 wheel) |
| Ballast | **none added**; the regulation ballast area still exists | the body must *itself* bring the car to 48.0 g (target 48.2 g) |

### 1.2 Regulations the body shape must respect (encoded as hard masks in Part 1)

T4.2 virtual cargo (a 52 × 58 × 8 mm solid that must be inside the body),
T5.x cartridge chamber and the **T5.5 3 mm wall** around it, the halo pocket and
visibility zones, the ballast area, **T7.9 wheel visibility zones**, wheel
keep-clear, T3.4/T3.5 width and height, the Model Block (223 × 65 × 50 mm), T3.7
1.5 mm track clearance, T8.2 and T8.5.1 at the nose.

### 1.3 The consequence of "no ballast" (the most important design fact)

The car must weigh 48.2 g whatever it looks like. Every gram that the printed
parts do not carry must come from the foam body. The foam is 0.163 g/cm³ and PA12
is 1.01 g/cm³ (6.2× denser), so a gram removed from a printed part costs about
**6 cm³ more body** and more frontal area. Under this rule, light parts are not
free and dense, low parts are useful. Every "save weight" result earlier in this
project was obtained with ballast and is invalid now.

---

## 2. What we have measured (evidence the decision rests on)

All CFD: ESI OpenFOAM v2412 on GitHub Actions, half car, every part in the flow,
rotating wheels, medium mesh unless stated.

| Finding | Numbers |
|---|---|
| Mesh convergence | same car: coarse 0.404 N, **medium 0.413 N, fine 0.410 N** (medium within 0.7 % of fine) |
| Run-to-run noise | 3–5 identical runs spread **0.18–1.0 %** in D20 |
| Where the drag is (final car) | front wheels 42 %, rear wheels 25 %, body 12 %, supports 7 %, front wing 7 %, halo 5 %, rear wing 3 % |
| Ceiling under the rules | if only the (un-fairable, T7.9–T7.11) wheels made drag: **1.055 s** at real thrust vs 1.097 s for today's car |
| Drag-adjoint body steps (level set) | **12 of 12** optimiser steps raised D20 (+2.7 % to +5 %), coarse and medium, 1 mm and 0.3 mm steps |
| One small adjoint step (0.3 mm, medium) | ranked correctly: descent −1.3 %, zero-size step −0.7 %, ascent +0.9 %, a real but tiny signal |
| Adjoint stability | medium converged; coarse **diverged** with the +6° front wing even with the stabilised (`cancel`) model |
| Carve depth (level set) | body drag not monotonic in how much was carved: 0.5 g reserve 0.389 N, 1.5 g 0.405 N, 3 g 0.394 N, 5 g 0.400 N, 8 g 0.400 N |
| Parametric loft, round 0 (old rules) | best of 13 valid designs **1.567 s** vs the carved car's 1.562 s; one "winner" was a non-converged solve showing −0.16 N body drag |
| Wheel supports | a thin strut cut its own drag 85 % but the car got **0.7 % worse**: the v2 support's rear **disc** shelters the rear wheel |
| Parts that worked (sweeps) | front wing +6° with an 8 mm 30° flap and 1 mm hubcap dome: **−2.6 % to −7 %** depending on the round; every one confirmed on repeats |

What this says, in one line: **shape changes that are smooth, local to the
parts, and tested with repeats have paid; free-form body changes steered by the
adjoint have not.**

---

## 3. The methods considered

For each: how it works, for, against, the evidence, and a verdict.

### 3.1 Level-set topology carve (the original Part 1)

*How:* the body is a signed-distance field on a 1 mm grid, starting as the full
block. Each iteration moves the surface with a velocity built from the mass
gradient and the CFD drag adjoint.

*For:* total shape freedom (it can grow or remove anything); produces
"generative-design" organic shapes; the rule masks plug straight in.

*Against:*
- It has no memory of design intent. Every cell moves independently, so steps
  are spiky: 12 of 12 drag steps got worse. A single step's true signal is about
  0.6 %, the same size as the noise.
- Its one working lever was **mass**. With no ballast, mass is set by sizing, so
  that lever is gone.
- Its shapes are blobby and hard to machine; the machining step then rewrites
  them anyway.
- Its outcome depends chaotically on setup (the carve-depth table above).

*Verdict:* **retire as the body designer.** Keep its grid, masks, machining step
and mesher as the *rules engine* every method uses.

### 3.2 Parametric loft, 3 stations (built 2026-09-27, `param_body.py`)

*How:* superellipse cross-sections at Ref A, the maximum section and the rear
face, blended smoothly; a second low, wide loft for the sidepods; an optional
floor plate. About 22 numbers.

*For:* every design is smooth and intentional; a small parameter space fits a
CFD-only search; easy to explain and to hand to CAD.

*Against:*
- Too few sections to be called organic: it can't make a coke-bottle waist, an
  undercut, a shaped engine cover or a sculpted underside.
- The rule masks and the machining step cut blocky patches into it (visible in
  the renders).
- Round 0 did not beat the carved car (1.567 s vs 1.562 s).

*Verdict:* **keep as the base layer**, not the whole answer.

### 3.3 Richer loft: 5–8 stations, spline blending (pure parametric)

*How:* the same as 3.2 with more sections and B-spline interpolation between them.

*For:* much more shape freedom with the same smoothness; still gradient-free;
easy to build (a small change to `param_body.py`).

*Against:*
- About 50–70 numbers. A CFD search at 16 cars per round needs hundreds of
  cars, i.e. days of GitHub time, for a reliable optimum.
- The added freedom is mostly along the car, not where drag is decided (around
  the wheels and the front of the sidepods).

*Verdict:* **use as the loft layer** (8 stations), but it needs a gradient to be
searchable.

### 3.4 Hybrid (proposed): parametric skeleton + rich loft + smooth sculpt modes, adjoint projected onto the modes

*How:*
1. **Skeleton:** every rule and assembly feature is fixed and not optimisable
   (Ref A/B, axles, chamber + T5.5 wall, cargo, halo pocket, ballast area, T7.9
   zones, support channels, nose joint).
2. **Loft:** 3.3's 8-station loft.
3. **Sculpt modes:** 20–40 smooth bump functions spread over the surface. Each
   is a Gaussian-like displacement a few mm across, with a strength the optimiser
   sets. They add controlled local freedom: shoulders, waists, the region behind
   the front wheels.
4. **Gradient:** one adjoint solve gives the drag sensitivity on the surface.
   Projecting it onto each mode gives dDrag/d(strength) for all modes at once.
   Each step is then **smooth by construction**, unlike the level set. The loft
   numbers get the same projection.
5. **Every candidate** goes through the rules engine, machining and mass sizing,
   then CFD. A step is accepted only if the measured race time improves by more
   than the noise.

*For:*
- It keeps design intent (the loft) and still allows organic local shaping.
- It gives the adjoint a fair second chance on smooth modes, which is the
  standard approach in industrial adjoint shape optimisation.
- It degrades gracefully: with the mode strengths at zero it is a loft.

*Against:*
- Unproven here. The adjoint's per-step signal was about 0.6 % against 0.2–1 %
  noise, and it diverged on the coarse mesh with the current wings.
- The machining and mass-sizing steps change the shape after the modes act, so
  the gradient is taken on a shape slightly different from the one milled
  (typically a sub-millimetre skin, but real).
- It is the most work of these options.

*Verdict:* **proposed**, with an explicit kill criterion (section 6).

### 3.5 Free-form deformation (FFD) lattice or OpenFOAM `volumetricBSplines`

*How:* put the body inside a lattice of control points; moving a point smoothly
deforms everything inside. ESI OpenFOAM's `adjointOptimisationFoam` has this
built in as `volumetricBSplines` design variables, with sensitivities computed
on the control points.

*For:* smooth and established. The OpenFOAM version gives consistent
sensitivities and morphs the CFD mesh without remeshing, which removes remeshing
noise.

*Against:*
- It deforms the **CFD mesh**, not our body. Every rule, machining and joint
  check would have to be redone on the morphed shape and pushed back into the
  grid, a two-way translation we don't have.
- Lattice points don't know the rules, so many moves will be illegal and get
  clipped.

*Verdict:* **not as the main method.** Mathematically it's close to 3.4's
sculpt modes, and 3.4 keeps the rules engine in charge. Revisit the OpenFOAM
lattice only as a way to cut remeshing noise.

### 3.6 NURBS surface modelling (direct control points, or CadQuery/OpenCascade)

*How:* build the body as exact CAD surfaces (B-rep); optimise the control points.

*For:* true CAD output (STEP), perfect for the team's drawings (T1.15, T7.12.2,
T8.1) and the CAM programme.

*Against:*
- Booleans with rule volumes and cutter-radius constraints on B-rep are fragile.
- Hundreds of control points need a gradient anyway (back to 3.4).
- A new dependency (OpenCascade) and a large rewrite.

*Verdict:* **no for optimisation; yes as the export format later.** Convert the
final body to STEP for CAD and CAM.

### 3.7 Machine-learning generative models (GAN/VAE, neural implicit shapes, surrogates)

*For:* fashionable; a trained surrogate could predict drag in milliseconds.

*Against:* no training data (we have ~100 CFD cars, not 10,000); a model that has
never seen the rules produces illegal cars; can't be explained to judges.

*Verdict:* **no.** A surrogate fitted to the CFD results *inside* 3.3/3.4's
search is fine (a Gaussian process over the loft parameters) and cheap.

### 3.8 Pure gradient-free search on everything (CMA-ES / Bayesian optimisation)

*For:* needs no adjoint; robust to noise if every car is run twice.

*Against:* cost grows with the number of parameters. Even at 22 it's 5+ rounds of
16; at 60 parameters it's weeks of runner time, and noise-driven.

*Verdict:* **for the loft layer only** (≤ 22 numbers), not the sculpt modes.

### 3.9 Designer in the loop (the team designs in CAD, the pipeline evaluates)

*How:* the team draws variants; Part 5 runs each through legality, machining,
mass and CFD, ranks them, and says what to change.

*For:* uses the team's judgement; every result is buildable by construction; it
matches how STEM Racing is judged (the design process is scored).

*Against:* slow turnaround; bounded by what the team thinks to try.

*Verdict:* **always available alongside.** The pipeline should accept a
team-drawn body STL as input (it nearly does) and report on it the same way.

### 3.10 Summary

| Method | Organic freedom | Steerable | Rule-safe | Machinable | Cost | Verdict |
|---|---|---|---|---|---|---|
| Level-set carve | very high | **no** (12/12 worse) | yes (masks) | after rewrite | high | retire as designer |
| 3-station loft | low | yes (search) | yes | yes | low | base layer |
| 8-station loft | medium | needs gradient/surrogate | yes | yes | medium | loft layer |
| **Hybrid: loft + sculpt modes + projected adjoint** | high, controlled | **to be proven** | yes | yes | high | **proposed** |
| FFD / volumetricBSplines | high | yes (built in) | no (mesh-side) | unknown | high | noise tool only |
| NURBS / B-rep | high | needs gradient | fragile | hard | very high | export only |
| ML generative | high | n/a | no | no | very high | no |
| Gradient-free on all | any | yes, slowly | yes | yes | very high | loft layer only |
| Designer in the loop | team's | human | yes | yes | team time | always, alongside |

---

## 4. Challenges (whatever method is chosen)

### 4.1 Physics and measurement
1. **Noise vs. signal.** Real body gains are ~0.5–3 % against 0.2–1 % noise.
   Every claimed gain needs a repeat (a paired base and variant in the same run).
2. **The wheels dominate** (67 % of drag) and can't be faired (T7.9–T7.11). The
   body's main aero job is managing flow onto and around the wheels (shown by the
   rear-disc result), not its own drag. The objective has to be the **whole-car**
   race time, never body drag.
3. **Steady, symmetric, half-car RANS at 20 m/s** can't see unsteady wakes or yaw.
   The medium mesh is within 0.7 % of fine, so resolution is not the limit;
   modelling is.
4. **Adjoint reliability:** converges on the medium mesh, diverges on coarse with
   the current wings. The hybrid method needs the medium adjoint (~30 min per
   step).
5. **Race model:** the thrust curve has ~60 % of the real impulse (rankings hold,
   absolute times don't); the centre-of-mass height term is a **placeholder**
   curve centred on 30 mm. With no ballast, the centre of mass now matters more
   and this term needs real data (track tests or a physics model).

### 4.2 Mass (no ballast)
6. **The mass loop:** body size depends on part masses, and part masses (plugs,
   keels) depend on the body. It needs a damped fixed point; a part that appears
   or disappears between passes (the nose did) makes it oscillate. Fixed by making
   the nose a separate part, but any new discrete feature can reintroduce this.
7. **Sizing by uniform skin offset** grows or thins the whole body equally, which
   is not how a designer would add mass (low and forward). A better sizing lever
   would be where the mass goes, and that's a design choice (section 7).
8. **Mass accuracy:** voxel mass at 1 mm is ±0.3 g; the target margin is 0.2 g.
   Final mass must be checked on the machined mesh, not the grid.

### 4.3 Manufacturing
9. **Cutter reach and stick-out are not modelled**: a 6.25 mm ball reaching 20 mm
   into a side feature may not be possible with the team's tool length or holder.
   It needs the tool's usable length as an input.
10. **Rule gap vs mandatory solid conflicts:** about 0.37 cm³ where a rule-required
    gap meets a mandatory solid (e.g. the ballast area's floor edges inside the
    cargo block) can't be cut with a ball end. It needs a flat end mill or hand
    finishing; this is reported, not solved.
11. **The halo pocket is enlarged** by the cutter radius at its corners. It must
    still hold the official halo within its fit tolerance (not checked).
12. **Joints:** exact-fit plugs, no glue gap, no print tolerance, no strength
    check. SLS PA12 needs ~0.1–0.2 mm clearance and the press fit on 3 mm stubs
    needs reaming. The **T7.13 100 g hang test** clearance between the wheel's
    inner corner and the body is not checked yet.
13. **Support channels split the foam:** a full-width channel from below at each
    axle leaves the foam joined only above the channel until the beam is glued.
    Handling strength in the milled-but-unassembled state is unknown.
14. **The rear wing can't be pocketed:** it sits over the T5.5 wall (3 mm of foam
    over the chamber), so it is surface-bonded.

### 4.4 Rules and scrutineering
15. **T7.9 zone dimensions** come from diagrams I don't have; they are encoded
    from the text. They need checking against the official figures.
16. **T4.2 cargo** partly inside printed plugs: "inside the car body" may be read
    strictly.
17. **Drawings:** T1.15, T7.12.2 and T8.1 require hatched identification of
    supports, nose and wing structures in the engineering drawings. The pipeline
    produces meshes, not drawings. A STEP export (3.6) is the bridge.

### 4.5 Compute and process
18. **GitHub free runners:** 20 jobs in parallel, 6 h per job, 4 cores. Medium
    CFD ≈ 30–35 min, adjoint ≈ 30 min, fine CFD ≈ 3.5 h. A 16-car round is ~40 min.
    A 20-step hybrid optimisation is ~20 h of wall time.
19. **Two geometry representations** (grid body, mesh parts) mean conversions and
    mesh booleans; each has failure modes (seen: non-watertight footprints, slivers).

---

## 5. The proposed plan (if the team agrees)

**Phase 0: lock the inputs (decisions, section 7).**

**Phase 1: rules engine and manufacturing, finished and tested (mostly built).**
- Machining for four setups: done.
- No-ballast mass sizing: done.
- Printed parts with joints (supports with drop-in channels and keels, nose shell
  with pocket, wing mount, tether guides): done except the nose-as-separate-part
  change and the PA12 wheel.
- The team's v2 support architecture as the reference: done.
- Add: glue/print clearances, T7.13 hang-test clearance, cutter stick-out limit,
  final mass on the machined mesh.
- *Success:* the reference car (team supports) and the parametric car both pass
  every check at 48.2 ± 0.1 g, with manufacturing files a person can open and
  make sense of.

**Phase 2: the loft layer.**
- Extend to 8 stations with spline blending.
- Gradient-free search: rounds of 16 with the best re-run each round, plus a
  surrogate over results.
- *Success:* beat the carved car (1.562 s) by more than the noise, confirmed on
  3 repeats.

**Phase 3: the sculpt layer, with a kill criterion.**
- 20–40 smooth modes; adjoint projected onto them; steps accepted only on a
  measured, repeated improvement.
- **Kill criterion:** if 4 consecutive accepted-step attempts fail to improve
  race time by more than the noise, stop. The loft result stands and the effort
  goes to wheels and supports instead.

**Phase 4: hand-off.**
- STEP export of the body and parts.
- A drawings pack skeleton (support/nose/wing identification).
- The machining report (unreachable corners, flat-end-mill areas).

---

## 6. Honest expectations

- The body is 12 % of drag. Even a 25 % body-drag cut is ~3 % of the car:
  about 5–8 ms at real thrust.
- The bigger levers are how the body shelters the wheels and the support shapes
  (the rear disc showed this). The hybrid's value is that its sculpt modes can
  act around the wheels.
- The sweep rounds on the wings found −2.6 % to −7 % with little risk. More part
  sweeps (support disc front and rear, beam section, nose shape) may pay faster
  than any body method.

---

## 7. Decisions needed from the team

1. **Wheels:** are `hardware_cad/front_wheel.stl` and `rear_wheel.stl` your
   current wheels (to print in PA12)? If not, should I build a parametric PA12
   wheel?
2. **Go / no-go on the hybrid** (3.4), or stop at the 8-station loft (3.3) and put
   the effort into part sweeps?
3. **Where the mass should go** when the body needs more: uniform skin (now),
   thicker lower body and floor (lower centre of mass), or denser printed parts?
4. **Tooling:** the 6.25 mm ball-end's usable length (stick-out), and whether a
   flat end mill is available for the few corners the ball can't cut.
5. **Tolerances:** the glue gap and print clearance for PA12 plugs, and how the
   3 mm stubs are sized for the bearing press fit (printed oversize and reamed?).
6. **Hang test (T7.13):** the required clearance between the wheel's inner corner
   and the body, from the regulation figure.
