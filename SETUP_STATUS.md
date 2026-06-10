# ATEC 2026 Local Setup Status

Last updated: 2026-05-21

## Directory Layout

- Workspace: `/home/1ctnltug/atec2026`
- Challenge project: `/home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge`
- Challenge zip: `/home/1ctnltug/atec2026/archives/ATEC2026_Simulation_Challenge.zip`
- Isaac Lab v2.3.2: `/home/1ctnltug/atec2026/IsaacLab`
- System Isaac Lab: `/opt/IsaacLab` via `/home/1ctnltug/Desktop/IsaacLab`
- System Isaac Sim: `/opt/IsaacSim`
- Deployment guide copy: `/home/1ctnltug/atec2026/notes/ATEC2026_DEPLOYMENT_GUIDE.md`

## Which Isaac Lab To Use

There are two Isaac Lab directories on this machine:

- `/opt/IsaacLab`: system image copy, version `2.2.0`.
- `/home/1ctnltug/atec2026/IsaacLab`: ATEC workspace copy, version `2.3.2`.

Use the workspace copy for ATEC:

```bash
/home/1ctnltug/atec2026/IsaacLab
```

## Conda Environments

### `atec2026`

This is the spec environment from the deployment guide:

- Python `3.12.13`
- Torch `2.7.1+cu128`
- Torchvision `0.22.1+cu128`
- Torchaudio `2.7.1+cu128`

Activation:

```bash
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026.sh
```

Use this for Python 3.12 package checks. Do not use it to launch this image's `/opt/IsaacSim`, because the installed Isaac Sim binary modules are Python 3.11 (`cp311`) builds.

### `atec2026-sim`

This is the practical simulation environment for the current image:

- Python `3.11.15`
- Uses `/opt/IsaacSim/setup_conda_env.sh`
- Isaac Lab v2.3.2 packages installed editable
- ATEC extension installed editable

Activation:

```bash
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh
```

Use this environment to run local simulation on this machine.

## Installed

- Miniconda: `/home/1ctnltug/miniconda3`
- Isaac Lab v2.3.2 cloned into workspace
- Isaac Sim linked from workspace Isaac Lab:

```text
/home/1ctnltug/atec2026/IsaacLab/_isaac_sim -> /opt/IsaacSim
```

- ATEC challenge zip extracted
- `atec_rl_lab` installed editable
- `scripts/list_envs.py` verified and listed 19 ATEC environments

## Verified Run Command

Headless Task A B2wPiper can start, create the scene, reset, and step with:

```bash
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh
python scripts/play_atec_task.py --task ATEC-TaskA-B2wPiper --headless --enable_cameras --disable_fabric --num_envs 1 --debug
```

Notes:

- `--enable_cameras` is required because the ATEC environment spawns cameras.
- `--disable_fabric` is required on this image because Isaac Sim bundles Warp `1.7.1`, while Isaac Lab v2.3.2's Fabric path expects `wp.transform_compose`.
- The default policy currently produces `total_episode_reward: 0.00`; this is expected until `demo/solution.py` is improved.

GUI run command:

```bash
source /home/1ctnltug/atec2026/scripts/env/activate_atec2026_sim.sh
python scripts/play_atec_task.py --task ATEC-TaskA-B2wPiper --enable_cameras --disable_fabric --num_envs 1
```

## Development Entry Point

Edit only:

```text
/home/1ctnltug/atec2026/ATEC2026_Simulation_Challenge/demo/solution.py
```

Submission shape:

```text
submission.zip/
├── solution.py
├── requirements.txt
└── models/
```
