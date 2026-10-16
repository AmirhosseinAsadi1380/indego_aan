# Indego Assist-as-Needed: Final Plan

Oct 14, 2026 · @Amirhossein

## How to use this plan

This plan is written so an AI agent (or a person) can carry it out. The goals are fixed; the methods and sources
are suggestions.

- **Fixed:** the overall goal, the three success criteria, and each phase's *done when* test.
- **Suggestions:** every method, dataset and code base named here, current as of October 2026. If one is missing,
  outdated, not licensed for our use, or does not get a phase to its *done when* test, search for a better one, use
  it, and add it to Sources with one line on why.
- Each phase lists "if stuck, search for" topics to start that search.
- Ask a human before: changing a goal or success criterion, collecting human-subject data beyond the sessions
  listed, or running anything on the real exo.

The human + Indego MuJoCo simulation already works. Start from it; do not rebuild it. First check its observation
set and control rate, because the deployable policy must use exactly the sensors the real Indego provides.

## Goal and success criteria

Build an Indego controller that keeps a person walking close to their own normal gait when their volition drops or
they fatigue, while giving only the assistance that is missing. Train and stress-test it in the existing human +
Indego simulation; use the lab only for calibration data and final validation.

The controller is judged on three things at once:

- **Gait stays normal.** Gait deviation from the participant's fresh baseline stays within their own fresh-session
  variability, under simulated and real fatigue.
- **Assistance is minimal.** Exo torque rises only when the human's contribution falls. With a fresh participant it
  stays near the transparent-mode level.
- **The human keeps working.** The human's share of hip and knee work does not drop more than the deficit requires.
  This is the slacking test.

> How the gait criterion is applied (decided 2026-10-14, project lead): the pass/fail deviation is the worst of the
> joints the Indego drives (hip and knee, both sides). Ankle deviation is reported alongside it but does not decide
> pass/fail, because the Indego has no ankle drive and the ankles would otherwise set the same floor for every
> controller. The criterion itself is unchanged.

Scope: able-bodied users and incomplete SCI (AIS C/D) for leg-level assistance. AIS A is handled as a separate
trunk/crutch variant (see Risks).

## Open data and tools

These are starting points, not a complete list. Public data covers healthy gait, EMG and fatigue well; SCI-in-exo
data is thin, so our own Indego logs fill that gap. DUO-GAIT is IMU-only, so a fatigue dataset with full kinematics
and EMG would be a useful find.

### Datasets

| Dataset | What it has | Use in this plan |
|---|---|---|
| AddBiomechanics Dataset | 273 subjects, 70+ h of mocap with force plates, scaled OpenSim models and joint torques | Main pretraining set for the human imitation policy; joint torques for the kinetic tracking term |
| Camargo et al. 2021 (Georgia Tech) | Treadmill at many speeds, level ground, ramps, stairs; mocap, force plates, EMG, IMU, goniometers | EMG check on muscle activations; speed range for volition sampling |
| Molinaro et al. 2024 task-agnostic exo dataset | 25 users, 28 activities, hip-knee exo sensor data with biological joint moments | Pretrain a joint-moment estimator from exo-like sensors; template for the student's sensor-to-state mapping |
| DUO-GAIT | 16 adults, 6-min walks before and after a muscle fatigue protocol, 9 IMUs, Borg ratings | Ground truth for how gait changes with fatigue; target for fatigue-model calibration |
| KIT Motion Dataset (via KINESIS) | 1.8 h of locomotion clips from AMASS; KINESIS converts them for MyoLeg | Fastest start for the imitation policy |
| Notre Dame EksoGT intent-change data | Onboard exo sensors, 3 able-bodied + 3 SCI users, speed-up / slow-down / no-change, crutches and walker | Test whether the student can detect intent changes from exo sensors alone |
| TU Delft incomplete-SCI foot placement | Walking data from people with incomplete SCI | Gait-variability ranges for AIS C/D human models |
| U. Michigan LocoLab data | Kinematics, kinetics, EMG across incline and speed; fatiguing tasks with a knee exo | Extra speed/incline coverage; fatigue-with-exo reference |

### Code

| Tool | Use in this plan |
|---|---|
| KINESIS | Motion-imitation RL on MyoLeg (80, 86 or 290 muscles) with downloadable pretrained checkpoints and a built-in 3CC-r fatigue option; starting point for the learned human |
| LocoMuJoCo | JAX implementations of PPO, GAIL, AMP and DeepMimic on MJX or the MjWarp backend; AMASS and LAFAN1 data retargeted for MuJoCo models; template for the Phase 0 fast sim |
| MyoSuite / MyoAssist | MyoSuite 3 runs the same env on CPU or on GPU (mjlab, MuJoCo Warp) with thousands of parallel envs, and its fatigue and motor-noise wrappers behave the same on both; MyoAssist builds on it for human + device training |
| 3CC-r / 4CCr fatigue models (Looft 2018 family, 4CCr) | Background for the fatigue model; port the KINESIS/MyoSuite 3CC-r code into our sim's vectorised muscle model |

## Phased plan

Seven phases, each written as a goal, a *done when* test, suggested starting sources, and what to search for if
those fall short. Phases 0 to 5 are simulation-only; the lab is needed only for the sessions in Lab experiments and
for Phase 6.

### Phase 0: Training throughput (only if needed)

- **Goal:** train policies on the human + Indego sim in days, not weeks.
- **Done when:** roughly 10^8 env steps run in a few days, and a fast training variant reproduces the full sim's
  joint trajectories under the same torque replay.
- **Start with:** MyoSuite 3's mjlab (MuJoCo Warp) GPU backend; LocoMuJoCo's MJX/MjWarp path; simplified cuffs and
  logging off for training only.
- **If stuck, search for:** GPU musculoskeletal simulation, MuJoCo Warp / mjlab examples, model reduction for RL.

### Phase 1: Realistic, adaptable human

- **Goal:** a muscle-driven human whose walking matches real gait and EMG across speeds, implemented as a learned
  policy so it can later re-optimize under a new objective (Phase 3).
- **Done when:** 60 s of walking without a fall from 0.6 to 1.4 m/s in transparent mode; leg-muscle on/off timing
  matches Camargo EMG; joint angles within normative ranges.
- **Start with:** pretrained KINESIS checkpoint (MyoLeg legs + abs, 86 muscles), fine-tuned on AddBiomechanics
  walking with a muscle-effort term; keep the sim's spinal reflex layer underneath.
- **If stuck, search for:** musculoskeletal motion imitation, MuscleMimic, AMP or DeepMimic with muscle actuators,
  EMG-validated muscle policies.

### Phase 2: Deficit models (fatigue, low volition, SCI)

- **Goal:** the sim human loses capacity (fatigue) or drive (low volition) gradually within a walk, with gait effects
  that match real people.
- **Done when:** without assistance, the degraded human's speed, step time and variability change in the same
  direction and roughly the same size as in fatigue datasets, and it falls only at the severe end of the range.
- **Start with:** 3CC-r per muscle (KINESIS option or MyoSuite 3 fatigue wrapper); drive scaling 0.3 to 1.0 with
  brief lapses and MyoSuite 3's motor-noise wrapper; DUO-GAIT for calibration; the sim's existing SCI weakness and
  spasticity profiles for AIS C/D.
- **If stuck, search for:** fatigue gait datasets with kinematics and EMG, central fatigue models, 4CCr,
  incomplete-SCI gait variability.

### Phase 3: The human's objective (IRL), so it co-adapts

- **Goal:** a human model that changes its own effort when assisted the way real users do, including slacking.
- **Done when:** the predicted change in human joint torques across assistance levels matches lab session C in
  direction and order of magnitude.
- **Start with:** a cost with weights on effort, deviation from own normal gait, and stability, fitted by bilevel
  search (inverse optimal control, as in the Waterloo work); adversarial IRL (GAIL/AIRL) from LocoMuJoCo as an
  alternative. Keep a naive (frozen) and a co-adapted human population.
- **If stuck, search for:** slacking in robot-assisted gait, human adaptation to exoskeleton assistance, inverse
  optimal control of gait, reward identifiability in IRL.

### Phase 4: Exo teacher policy (privileged)

- **Goal:** an assist-as-needed policy that, knowing the true deficit, meets all three success criteria.
- **Done when:** it beats the baseline Indego FSM on all three criteria across the deficit range, against both naive
  and co-adapted humans.
- **Start with:** PPO with privileged observations and the reward, observations and randomization in Exo policy
  specification; deficits drifting within each episode.
- **If stuck, search for:** assist-as-needed exoskeleton control, multi-agent human-device RL (MyoAssist),
  privileged teacher-student training for locomotion.

### Phase 5: Deployable student policy

- **Goal:** the teacher's behavior using only real Indego sensors, plus an interpretable estimate of the deficit.
- **Done when:** within a small margin of the teacher on the success criteria in the full-fidelity sim, and its
  fatigue/drive estimate tracks the true values.
- **Start with:** DAgger distillation into a TCN or GRU over sensor history; an auxiliary head for fatigue, drive
  and human joint moment, pretrained on the Molinaro exo dataset; intent-change sanity check on the Notre Dame EksoGT
  data.
- **If stuck, search for:** teacher-student distillation for legged robots, joint moment estimation from exoskeleton
  sensors, fatigue detection from IMU gait.

### Phase 6: Lab validation on the Indego

- **Goal:** show on the real device that gait stays normal under fatigue and low volition, with minimal assistance
  and no slacking.
- **Done when:** the student passes the validation table against transparent mode and the FSM.
- **Start with:** staged bring-up (transparency, then capped, then full assistance) and sessions D and E.

## Exo policy specification

The reward trades off normal gait against minimal assistance and human participation; the participation weight is
the main dial and should stay tunable, so a therapist can choose to challenge or support.

$$r_t = w_g\, r_{\text{gait}} + w_s\, r_{\text{stab}} - w_a \lVert \tau_{\text{exo}} \rVert^2 + w_p\, r_{\text{part}} - w_j \lVert \dot{\tau}_{\text{exo}} \rVert^2$$

| Term | Definition | Notes |
|---|---|---|
| r_gait | AMP-style discriminator score against the participant's fresh-baseline gait (sim: the Phase 1 human at full drive) | Lets a tired person walk slower with a normal pattern; penalises foot drag and asymmetry |
| r_stab | Margin of stability (extrapolated COM inside base of support, crutch tips included) | Hard termination on falls |
| exo torque | Squared hip and knee exo torque | Keeps assistance minimal when the human is fresh |
| r_part | Human's share of hip + knee positive work, penalised when it falls below what the current deficit allows | Stops the policy from simply taking over |
| torque rate | Squared rate of change of exo torque | Smoothness; also respects the drive's current-loop limits |

### Observations

| Policy | Sees |
|---|---|
| Teacher (sim only) | Indego sensors + true fatigue state per muscle group, drive scale, human joint torques, gait phase |
| Student (deployable) | History window of Indego sensors only: joint encoders, firmware velocity, motor current, thigh and hip IMUs, optional trunk IMU and crutch force |

### Randomization (training sim)

| Factor | Range | Source of the range |
|---|---|---|
| Fatigue rate per muscle group | Calibrated band around the DUO-GAIT fit | Phase 2 |
| Volitional drive scale | 0.3 to 1.0, drifting within an episode, plus brief lapses | Phase 2 |
| Walking speed command | Participant's slow to fast range | Camargo speed set; our own sessions |
| Human population | Naive vs co-adapted, several seeds each | Phase 3 |
| Body mass, segment lengths, cuff stiffness, axis misalignment | Measured ranges across our participants | Our participants' measurements |
| Drive friction, delay, torque constant | Bench-identified spread | Bench sysid (session A) |
| Sensor noise, latency, dropped samples | Measured on our Indego | Measured on our Indego |

## Lab experiments and validation metrics

Five short data collections are enough to calibrate the sim and fit the IRL step; session C is the one that cannot
be skipped, because open data has no Indego at multiple assistance levels.

| Session | Who | Protocol | Feeds |
|---|---|---|---|
| A. Bench sysid | Exo only | Actuator identification (chirp, step and constant-speed tests) with the exo on a stand | Drive model, Phase 0 |
| B. Fresh baseline | Each participant | Transparent mode, 3 speeds, 2 min each; exo logs, EMG, mocap if available | Personal "normal" for r_gait; replay check of the sim |
| C. Assistance levels | Able-bodied + AIS C/D | Same simple hip/knee assist profile at zero, low and medium gain, randomized order | IRL fit of the human's cost, Phase 3 |
| D. Fatigue | Able-bodied, then AIS C/D if safe | Walk, fatiguing protocol (e.g. weighted-vest sit-to-stand), walk again; Borg ratings, EMG median frequency | Fatigue-model check; estimator ground truth |
| E. Volition drop | Able-bodied | Instructed "relax your legs" segments during walking, cued by the experimenter | Labelled drive drops for the student's estimator |

Existing FSM session logs from SCI users are also used to calibrate the arm and crutch model before any SCI walking
results are trusted.

### Validation (Phase 6)

Compare three conditions in sessions D and E: transparent mode, the baseline FSM, and the student policy.

| Metric | Measures | Pass if |
|---|---|---|
| Joint-angle deviation from fresh baseline | Gait stays normal | Within the participant's fresh-session variability band |
| Speed and step-time variability | Gait stays normal | No larger drop than in fresh transparent walking |
| Exo torque and positive work | Assistance is minimal | Near transparent level when fresh; rises with fatigue |
| Human share of joint work (inverse dynamics or model-based) | Human keeps working | Falls no more than the deficit explains |
| Estimator output vs Borg, EMG median frequency, cued relax segments | Deficit detection | Rises with fatigue; flags cued drops |
| Falls, near-falls, cuff force peaks | Safety | Zero falls; cuff forces within the comfort limit we set |

## Risks, scope and schedule

The biggest risk is Phase 1: if the learned human cannot walk with the Indego attached, nothing downstream is
meaningful, so it gets the most time and a fallback.

| Risk | Fallback |
|---|---|
| Learned human falls once the exo is attached | Curriculum: imitation without exo, then exo with zero mass, then full mass; start from the 80-muscle legs model before adding abs |
| Training sim too slow even after Phase 0 | Many parallel CPU envs on a cluster; drop the spinal layer to control rate in training only |
| IRL weights not identifiable from our data | Sweep the effort weight by hand and require the exo policy to pass across the whole sweep |
| 3CC-r fatigue not validated for walking | Calibrate against DUO-GAIT and session D; treat fatigue rates as a randomized range, not a point value |
| Human work share hard to measure in the lab | Mocap + inverse dynamics in sessions B to D; model-based estimate from exo torques otherwise |
| Sim-to-real gap in the student | Staged bring-up with torque caps; the auxiliary estimator is checked against Borg and EMG before assistance is uncapped |

**Scope for AIS A.** There is no leg volition to lose, so leg-level assist-as-needed does not apply. The same
teacher-student machinery can instead detect arm and trunk fatigue (rising crutch load, changing trunk lean) and
adapt step timing and length. Treat this as a separate branch after Phase 5.

**Compute.** One modern GPU for the MJX/Warp training sim (MuscleMimic runs simulation and learning on a single
GPU), plus CPU cores for full-fidelity evaluation runs.

**Schedule.** Rough estimate for one researcher full time, about 11 months; session C should be collected during
Phase 1 so Phase 3 is not blocked. Each lab session feeds the phase on its row; C is used in Phase 3 but should be
collected during Phase 1.

## Sources

- AddBiomechanics Dataset paper (Werling et al.)
- Stanford FAIR Center datasets
- Camargo et al. dataset (Mendeley, part 1)
- Molinaro et al. dataset (Georgia Tech repository)
- DUO-GAIT (Scientific Data)
- Notre Dame EksoGT intent-change dataset (IEEE DataPort)
- U. Twente / TU Delft wearable robotics open data
- U. Michigan LocoLab data
- KINESIS paper
- MuscleMimic (CVPR 2026 workshop slides)
- LocoMuJoCo paper
- MyoSuite paper
- 3CC-r parameter study (J. Biomech. 2022)
- Four-compartment fatigue model (Front. Physiol. 2024)
- Recovering optimal cost functions for natural walking (Waterloo)
- Musculoskeletal motion imitation for personalized exo control
- Our human + Indego MuJoCo simulation and its documentation
