# KINESIS on this machine (Phase 1: a pretrained human)

KINESIS (github.com/amathislab/Kinesis, BSD-3) with its pretrained legs model (Hugging Face
`amathislab/kinesis-models`), run on CPU on Windows. Kept outside this repository: the checkout at
`%USERPROFILE%\kinesis`, the AMASS / SMPL downloads in `%USERPROFILE%\kinesis_downloads` (their licences do not
allow redistribution).

## Setup (Windows, no admin rights)

```bash
python -m pip install uv                      # any existing Python
python -m uv python install 3.8
git clone https://github.com/amathislab/Kinesis.git kinesis && cd kinesis
python -m uv venv --python 3.8 .venv
python -m uv pip install --python .venv/Scripts/python.exe "torch==1.13.1" --index-url https://download.pytorch.org/whl/cpu
grep -v -E "^#|^clip|^ipdb|^chumpy|^smplx" requirements.txt > req.txt     # CLIP: only for text-to-motion
python -m uv pip install --python .venv/Scripts/python.exe -r req.txt pip huggingface_hub
python -m uv pip install --python .venv/Scripts/python.exe --no-build-isolation "chumpy==0.70"
python -m uv pip install --python .venv/Scripts/python.exe "smplx @ git+https://github.com/ZhengyiLuo/smplx.git@a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813"
.venv/Scripts/python src/utils/download_assets.py --branch kinesis-2.0
.venv/Scripts/python -c "from huggingface_hub import snapshot_download; snapshot_download('amathislab/kinesis-models', local_dir='data/trained_models', allow_patterns=['legs/*'])"
```

SMPL: `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` from SMPL v1.1.0 for Python -> `data/smpl/SMPL_NEUTRAL.pkl`.

Local patches (Windows): `src/KinesisCore/kinesis_core.py`, `load_motions` - skip
`mp.set_sharing_strategy("file_descriptor")` (Linux only) and load motions in one process on win32, as KINESIS already
does on macOS; `src/utils/convert_kit.py` - split clip paths with `os.sep` (it assumed "/"); `src/agents/agent.py` - the
`/dev/shm` housekeeping only where it exists. Training still does not run on Windows (the sampler's worker processes
crash under Windows' spawn start); evaluation does. Train under Linux (WSL2 or a Linux machine).

Check (2026-10-15): KIT test set, 108 clips, `run.initial_pose_file=data/initial_pose/legs/initial_pose_test.pkl`,
`run.num_motions=108`: 88.9 % success, 47.2 mm MPJPE, 94.2 % frame coverage (paper, Legs-PD: 91.4 +- 1.8 %, 48.8 mm,
96.4 %).

## Motions

* `convert_amass.py <amass_subset_dir> <out.pkl> [filter] [exclude]` - any AMASS subset (SMPL+H G) to a KINESIS
  motion dictionary (KINESIS's KIT processing, Windows paths).
* `initial_poses.py <motion_dict.pkl> <out.pkl>` - the starting pose of each clip. Needed for clips outside KINESIS's
  own sets: without it the evaluation reuses the first clip's pose and position for every later clip.

## Evaluate

```bash
.venv/Scripts/python src/run.py --config-name config_legs.yaml exp_name=kinesis-moe-imitation epoch=-1 \
    learning.actor_type=moe run=eval_run_legs run.headless=True run.motion_file=<motion_dict.pkl> \
    run.initial_pose_file=<initial_poses.pkl> env.termination_distance=0.5 run.num_motions=<n> no_log=True
```

KINESIS's evaluation is deterministic (one clip = one sample) and judges only the positions of pelvis, knees, ankles
and toes (mean distance to the reference). Scripts that load motions must keep their code under
`if __name__ == "__main__":` on Windows: the motion library starts a helper process, which re-imports the script.
The motion library's body rotations (`xquat`, `root_rot`) are (w, x, y, z) quaternions; its angular velocities are
in the world frame.

## The exo, long walks and checks (fine-tuning)

* `kinesis_exo.py` - the Indego folded into KINESIS's model (mass, drive inertia / friction, springs for the
  frontal-plane hip and the AFO; the AFO spring acts about indego_sim's 5 deg dorsiflexion neutral).
* `exo_kinesis.py` - KINESIS's own `src/run.py` (training or evaluation) with that exo per episode (fixed or
  randomised), optional deficits (drive, fatigue) and an optional pelvis tracking reward (`--pelvis-reward`).
* `long_reference.py`, `build_training_set.py`, `merge_motions.py` - minute-long walking references from KIT clips,
  the training set with them, and one evaluation set of all of them; `long_eval_summary.py` summarises its log.
* `gait_check.py` - pelvis yaw / pitch / roll and hip ranges of a policy in KINESIS's simulation (what the position
  metrics do not see).
* `extract_experts.py`, `build_moe.py` - split the mixture of experts for fine-tuning and put it back together.
* `policy_server.py` - serves a policy to indego_sim (`indego_aan/kinesis_human.py`, `scripts/kinesis_walk.py`).

## Walking in indego_sim

```bash
python scripts/kinesis_walk.py --exo --rigid-trunk --kinesis-dir <kinesis> --model-dir <policy folder> \
    --motion data/long_walks/long_walk_0.8.pkl --pose data/long_walks/initial_pose_long_walk_0.8.pkl \
    --exo-set hip_abd_stiffness=0,hip_rot_stiffness=0 --T 60 [--transparent] [--record trial.npz]
```

`policy_server.py` runs the policy in KINESIS's own environment (Python 3.8) and serves it to indego_sim (Python
3.12) at 30 Hz: the person's state is mirrored onto KINESIS's model, KINESIS computes its observation against the
reference clip, and indego_sim turns the 80 returned target muscle lengths into activations with KINESIS's muscle PD
law at 150 Hz. The policy only walks under the integrator it was trained with (explicit Euler) and with a stiff
trunk (`--rigid-trunk`, KINESIS's torso is rigid). The run prints the pelvis and hip ranges (`waterloo.pelvis_ranges`).

## Fine-tuning with the exo (Linux)

KINESIS training does not run on Windows. From the KINESIS checkout, with KIT converted by KINESIS's
`src/utils/convert_kit.py`:

```bash
T=<this folder>
python $T/extract_experts.py data/trained_models/legs/experts_from_moe -1.0      # trainable copies of the 3 experts
python $T/long_reference.py data/long_walks 0.5,0.6,0.7,0.8,0.9,1.0 65 3          # minute-long walking references
for f in data/long_walks/long_walk_*.pkl; do
    python $T/initial_poses.py $f data/long_walks/initial_pose_$(basename $f); done
python $T/build_training_set.py data/long_walks data/train_long.pkl data/initial_pose/legs/initial_pose_train_long.pkl 6
python $T/merge_motions.py data/long_walks/eval_all.pkl data/long_walks/initial_pose_eval_all.pkl data/long_walks/long_walk_*.pkl

# each expert i = 0, 1, 2: copy data/trained_models/legs/experts_from_moe/expert_$i/model.pth into runs/exo_expert_$i, then
python $T/exo_kinesis.py --exo-mode random \
    --exo-ranges "kabd=0:600,krot=0:150,kank=0:150,ank0=0:10,mass=0.5:1.2,friction=0:1,p_none=0.2" \
    --deficit-ranges "p_drive=0.25,drive=0.4:0.9,drop_at=0:1,ramp=0.2:1,p_fatigue=0.25,fatigue=0:0.5" \
    --pelvis-reward "w_rot=0.15,k_rot=0.2,w_yaw=0.15,k_yaw=5" \
    --config-name config_legs.yaml exp_name=exo_expert_$i hydra.run.dir=runs/exo_expert_$i epoch=-1 \
    learning.actor_type=lattice run=train_run_legs run.headless=True run.num_threads=16 learning.max_epoch=<n> \
    run.motion_file=data/train_long.pkl run.initial_pose_file=data/initial_pose/legs/initial_pose_train_long.pkl \
    run.num_motions=5000 no_log=True

python $T/build_moe.py data/trained_models/legs/kinesis-moe-imitation/model.pth runs/exo_expert_ runs/exo_moe/model.pth
```

Evaluate: KINESIS's evaluation (above) through `exo_kinesis.py` with `--exo-mode fixed --exo-kabd ... ` and optional
`--deficit-ranges`, on the KIT test set or the merged long references (`long_eval_summary.py` summarises the log);
`gait_check.py` reports the pelvis and hip motion. One epoch is 51,200 simulation steps (~1-2 min on a 28-thread
CPU with 16 workers); 250 epochs per expert take ~5.5 h.
