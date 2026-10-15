# BDTF — code for review

BDTF (bias-aware dual-support target fusion) is an offline reinforcement
learning method. This folder contains its complete implementation, the offline
datasets, and the environments for the four tasks of the paper:

| Task name (`--task`) | Dataset | Evaluation environment |
|---|---|---|
| `d4rl-halfcheetah` | D4RL `halfcheetah-medium-replay-v2` | Gym MuJoCo `HalfCheetah-v3` |
| `d4rl-hopper` | D4RL `hopper-medium-replay-v2` | Gym MuJoCo `Hopper-v3` |
| `d4rl-walker2d` | D4RL `walker2d-medium-replay-v2` | Gym MuJoCo `Walker2d-v3` |
| `rolling` | steel-strip flatness data logged on a finishing mill | flatness simulator in `rolling_env/` |

Selecting a task selects its dataset, environment, training and evaluation
protocol, and all BDTF hyper-parameters. No other setting is required.

## Contents

```
.
├── train.py               training and evaluation (entry point)
├── check_setup.py         installation check
├── environment.yml        conda environment with the package versions of the paper
├── ENVIRONMENT.md         system used for the paper (OS, GPU, driver, CUDA)
├── collect_env.py         records the environment of the current machine (into my_environment/)
├── configs/
│   ├── default.yaml       hyper-parameters shared by all tasks
│   └── <task>.yaml        protocol and task-specific hyper-parameters
├── bdtf/
│   ├── agent.py           the BDTF algorithm
│   ├── networks.py        critic ensemble, actor, conditional VAE behaviour model
│   ├── mmd.py             state-conditional MMD
│   ├── buffer.py          offline dataset with per-head bootstrap masks
│   ├── evaluation.py      closed-loop evaluation and flatness-control metrics
│   ├── config.py
│   └── envs/
│       ├── d4rl.py        D4RL datasets and Gym MuJoCo environments
│       └── rolling.py     strip-flatness simulator and dataset
├── rolling_env/
│   ├── flatness_simulator.pt   dynamics model of the flatness loop
│   └── scaler.json             standardisation of the mill signals
└── data/
    ├── d4rl/              the three D4RL HDF5 files
    └── rolling/           rolling_offline.pickle (3412 transitions)
```

## 1. Installation

All package versions are given in `environment.yml`; they are the versions used
for the paper (Linux, Python 3.9, PyTorch 2.1.2 with CUDA 12.1; the system is
described in `ENVIRONMENT.md`).

```bash
conda env create -f environment.yml
conda activate bdtf
```

This is all the rolling task needs. The D4RL tasks additionally need the
MuJoCo 2.1 binaries for the Gym `-v3` environments (the D4RL v2 datasets were
collected with the `-v3` dynamics):

```bash
sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf build-essential
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xzf mujoco210-linux-x86_64.tar.gz && cd -
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
```

The first `import mujoco_py` compiles a C extension (one to three minutes).
Warnings from Gym that it is unmaintained and that the `-v3` environments are
out of date are expected.

Check the installation:

```bash
python check_setup.py            # rolling task and the three D4RL tasks
python check_setup.py --rolling  # rolling task only
```

For the rolling task the check should print a score of `0.00` (it re-measures
the zero point of the score scale). The first use of each D4RL dataset creates a
cached `*.qlearning.npz` file next to it.

## 2. Running

All commands are run from this folder.

```bash
python train.py --task d4rl-halfcheetah --seed 10
python train.py --task d4rl-hopper      --seed 10
python train.py --task d4rl-walker2d    --seed 10
python train.py --task rolling          --seed 20
```

Several seeds in one call, separated by `-`.

### Seeds of the paper

| Tasks | Seeds |
|---|---|
| D4RL (each of the three tasks) | `10-100-1000-10000-100000` |
| rolling | `20-200-2000-20000-200000-30-300-3000-30000-300000-40-400-4000-40000-400000-50-500-5000-50000-500000` |

For example, all runs of the paper on hopper and on the rolling task:

```bash
python train.py --task d4rl-hopper --seed 10-100-1000-10000-100000
python train.py --task rolling --seed 20-200-2000-20000-200000-30-300-3000-30000-300000-40-400-4000-40000-400000-50-500-5000-50000-500000
```

The runs of one call are executed one after another; to use several GPUs, start
one call per seed (or per group of seeds) with `CUDA_VISIBLE_DEVICES` set.

Options:

| Option | Meaning |
|---|---|
| `--device auto\|cuda\|cpu` | default `auto`: GPU if available |
| `--out DIR` | output folder, default `results/` |
| `--quick` | short functional check (2000 steps, 2 evaluation episodes); **not** the paper's protocol |

A quick functional check of the whole pipeline takes about two minutes on a CPU:

```bash
python train.py --task rolling --seed 20 --quick
```

### Protocol and run time

| Task | Gradient steps | Behaviour-model pre-training | Evaluation | Run time per seed |
|---|---|---|---|---|
| D4RL (each) | 500,000 | 40,000 steps | every 10,000 steps, 10 episodes of up to 1000 steps | about 1.5–2 h on an RTX 4090 |
| rolling | 30,000 | 8,000 steps | every 1,000 steps, 50 episodes of 100 steps | about 5 min on an RTX 4090 |

## 3. Output

Each run writes `results/<task>-seed<seed>-<time>/`:

| File | Content |
|---|---|
| `config.json` | the full configuration used (task, protocol, hyper-parameters) |
| `progress.csv` | one row per evaluation: normalized score, return, training diagnostics (fusion weight `train_w_mean`, MMD, Lagrange multiplier, actor learning rate); for `rolling` also flatness RMSE, maximum deviation, in-tolerance fraction and move variation |
| `summary.json` | `final_score_last10` |
| `models/final.pt` | the trained networks |

**Final score.** A run is scored by the mean normalized score of its last ten
evaluations (`final_score_last10`, also printed at the end of the run). The
paper reports the mean and the standard deviation of this score over the seeds
listed above.

