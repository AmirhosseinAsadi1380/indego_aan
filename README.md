# indego_aan

Assist-as-needed control for the Indego exoskeleton: a controller that keeps a person walking close to their own
normal gait when their volition drops or they fatigue, while giving only the assistance that is missing. Trained and
stress-tested in the [indego_sim](https://github.com/AmirhosseinAsadi1380/indego_sim) human + Indego simulation.

- Plan (goals, success criteria, phases): [docs/PLAN.md](docs/PLAN.md)
- Where it stands: [docs/STATUS.md](docs/STATUS.md)
- Sensors the deployable policy may use: [docs/SENSORS.md](docs/SENSORS.md)
- Sources checked and used: [docs/SOURCES.md](docs/SOURCES.md)

## At a glance

* **Deficit models** (Phase 2): volitional drive loss and 3CC-r muscle fatigue on the simulated person (T12 AIS D,
  sagittal plane), who degrades gracefully - no falls down to drive 0.4.
* **Controller** (Phase 4): assist-as-needed toward the person's own fresh gait. Essentially no assistance when fresh;
  no falls with drive loss to 0.4, fatigue or pre-fatigue; the person keeps doing 83-96 % of the hip/knee work. Four
  PPO teachers and a CMA-tuned privileged version did not beat it; nothing tried survives drive 0.3.
* **Deficit estimate** (Phase 5): the controller's adapted gains estimate the person's lost capacity from Indego
  sensors alone (R^2 0.85 on held-out runs, 2-3 strides behind a drive drop).
* **Learned human** (Phase 1): the pretrained KINESIS legs policy runs here and reproduces its published test result;
  it covers slow walking only, so normal speeds need fine-tuning (on Linux).

## Setup

```bash
pip install -e ../indego_sim            # plus indego_sim's own setup (MyoLeg model)
pip install -r requirements.txt
python -m pytest -q tests
```

The scripts start one worker per physical core with single-threaded BLAS, capped by free memory
(`indego_aan/resources.py`). DUO-GAIT data: download `processed.zip` from Zenodo (10.5281/zenodo.7415758) into
`data/duo_gait/` and unzip it there.

## Contents

| | |
|---|---|
| `indego_aan/scenario.py` | the base scenario: a person walking on their own in the Indego (sagittal model) |
| `indego_aan/humans.py` | simulated people: walking-parameter sets (a walker robust to its own deficits) |
| `indego_aan/deficits.py` | low volitional drive and 3CC-r muscle fatigue, attached to the simulated person |
| `indego_aan/rollout.py` | instrumented rollouts (human and exo joint torques, work, gait events, deficit state) |
| `indego_aan/metrics.py` | the three success criteria: gait normality, minimal assistance, human participation |
| `indego_aan/controllers.py` | assist-as-needed toward the person's own fresh gait (baseline for the learned policy) |
| `indego_aan/env.py` | Phase 4 teacher environment: privileged observations, the plan's reward, deficit randomisation |
| `scripts/throughput.py` | Phase 0: control steps per second of the full simulation |
| `scripts/deficit_sweep.py` | Phase 2: how the unassisted person's gait changes with deficits |
| `scripts/tune_robust_walk.py` | a person who copes with their own deficits (naive human population) |
| `scripts/duo_gait.py`, `scripts/calibrate_fatigue.py` | Phase 2: DUO-GAIT fatigue effects and the matching pre-fatigue state |
| `scripts/fresh_reference.py` | the person's fresh gait curves (exo encoders and human joints) |
| `scripts/baselines.py` | existing controllers (and the teacher) on the success criteria across deficits |
| `scripts/train_teacher.py` | Phase 4: PPO teacher training (Stable-Baselines3, CPU) |
| `scripts/tune_teacher_aan.py` | Phase 4 teacher: deficit-scheduled assist-as-needed tuned by CMA-ES on fixed episodes |
| `scripts/deficit_estimate.py` | Phase 5: how well the deployable AAN's gains estimate the person's lost capacity |
| `scripts/kinesis/` | Phase 1: running the pretrained KINESIS human here (setup, AMASS conversion, initial poses) |
| `scripts/supervise_training.py` | runs the training on a shared PC: pauses it when memory runs low, resumes from the last save |

Trained-policy binaries are not in the repository (their settings, learning curves and evaluations are, under
`results/`); third-party datasets (DUO-GAIT, AMASS, SMPL) are downloaded locally and never committed.
