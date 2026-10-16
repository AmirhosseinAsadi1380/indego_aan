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
