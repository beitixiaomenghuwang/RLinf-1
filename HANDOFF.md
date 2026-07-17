# Pi0.5 + GSE Multi-Task RL Handoff

Last updated: 2026-07-17

This document is the current handoff for continuing the Pi0.5 + GSE work in a
new coding conversation. Read it before changing code or launching a long run.

## 1. Research objective and current hypothesis

The goal is to improve one MetaWorld MT50 VLA policy on many tasks at once with
parameter-efficient RL post-training, while reducing multi-task interference and
retaining the SFT policy as an exact initialization.

The current method is:

1. Load RLinf's fully supervised Pi0.5 MetaWorld MT50 checkpoint.
2. Insert zero-output GSE residual adapters into the Pi0.5 action expert.
3. Freeze the pretrained base policy.
4. Use RLinf's Flow-SDE/Flow-Noise PPO path to update GSE, its router, and the
   value head.
5. Compare against the raw SFT policy, official full-parameter PPO, and a
   parameter-matched plain LoRA PPO baseline under matched data and evaluation
   budgets.

This is not a second GSE-SFT stage and does not decompose an SFT weight delta.
Orthogonal A initialization plus zero B initialization makes the initial GSE
policy exactly equal to the loaded SFT policy. RL learns only a residual.

The same-pipeline, fixed-reset SFT/GSE evaluation is now complete. The latest
10-step GSE checkpoint improves aggregate success and worst-10 success, while
reducing the number of zero-success tasks from seven to five. This establishes a
short-run multi-task signal, but one seed with only 10/11 trials per task is not
yet a generalization or statistically stable balance claim.

## 2. Repositories, image, and assets

Local development repository:

```text
/home/caslx/Robotics/RLinf
```

Original GSE/MoORE reference repository:

```text
/home/caslx/Robotics/VLA-GSE
```

Relevant reference files:

```text
VLA_GSE/training/train_gse.py
VLA_GSE/training/train_moore.py
VLA_GSE/gse_peft/gse/layer.py
```

Local MetaWorld SFT checkpoint:

```text
/home/caslx/Model/RLinf-Pi05-MetaWorld-SFT
```

Docker image:

```text
rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
```

Activate the OpenPI environment inside Docker with:

```bash
source switch_env openpi
```

The target server has 8 x A100 80 GB. The latest GSE run used about
`53067 MiB / 81920 MiB` per occupied GPU, so memory is currently not the main
constraint.

## 3. Implemented method

### 3.1 GSE core and OpenPI integration

Core implementation:

```text
rlinf/models/peft/gse/
```

OpenPI integration:

```text
rlinf/models/embodiment/openpi/gse.py
rlinf/models/embodiment/openpi/__init__.py
```

Primary configuration:

```text
examples/embodiment/config/metaworld_50_ppo_openpi_pi05_gse.yaml
```

Default GSE architecture:

- total rank 64 per wrapped linear layer;
- 8 experts: 2 generalized and 6 specialized;
- specialized top-k 2;
- sequence-level mean-pooled routing;
- orthogonal A subspaces and zero-initialized B matrices;
- generalized experts are always active;
- specialized experts use a learned sparse router.

GSE is currently injected only into the 18-layer Pi0.5 action expert. The seven
wrapped projections per layer are `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`, for 126 adapters in total.

The loaded base policy is frozen. GSE and the RL value head remain trainable.
GSE and RLinf's existing LoRA mode are mutually exclusive.

### 3.2 FSDP constraints

Each complete `GSEAdapter` is grouped as one trainable FSDP unit. The working
MetaWorld configuration uses:

```yaml
actor:
  fsdp_config:
    sharding_strategy: "no_shard"
    use_orig_params: False
```

Do not change these casually. Independent wrapping of every expert leaf created
about 2,142 FSDP children. PyTorch 2.6 with `NO_SHARD` and
`use_orig_params: True` failed during optimizer writeback after the first Adam
step. The current grouping produces 247 FSDP modules and has completed repeated
PPO optimizer steps.

### 3.3 Multi-task observability and objectives

Implemented:

- router entropy and normalized entropy;
- expert selection/probability metrics;
- count-weighted task-conditioned expert selection/probability metrics, including
  normalized task/expert mutual information and Jensen-Shannon divergence;
- load-balance and orthogonality diagnostics;
- optional load-balance and orthogonality auxiliary actor losses;
- MT50 per-task success, macro success, worst-5/worst-10 summaries, and threshold
  counts;
- balanced train reset sampling and fixed balanced eval reset sampling;
- CUDA-safe metric reduction;
- checkpoint retention;
- eval-only runner support.

The first balanced pilot used monitoring-only auxiliary losses. Keep
`lambda_balance=0` and `lambda_orth=0` until paired evaluation and
task-conditioned router measurements justify changing the objective.

Uniform aggregate expert usage is evidence against global router collapse, but it
does not prove task specialization. Task-conditioned expert utilization is now
logged during PPO recomputation and should be used for that judgment.

## 4. Important commits

The main implementation history is:

```text
a060eb0e  GSE core
92a7efc4  OpenPI/Pi0.5 GSE integration
956ae53d  PPO auxiliary loss and router metrics
1f56f2e9  MetaWorld per-task success metrics
fd292f42  CUDA metric conversion fix
0d077caa  host-backed Ray spill/temp storage and video reduction
65035206  bounded checkpoint retention
b4ef12cb  balanced multi-task reset sampling
45f532b6  balanced reset-pool padding per worker
a6144d37  functional eval-only runner
dcac37c1  skip actor-to-rollout weight sync in eval-only mode
18f0c0e4  task-conditioned GSE router utilization
7cac562f  retain per-sample routing assignments only when requested
d292f580  parallel Pi0.5 rollout profiles and routing validation
39ccbf3b  validated batch16 rollout and actor budget
8646d1a0  high-memory Pi0.5 rollout and actor profiles
c798a718  reproducible rollout seeds and multi-seed evaluation summaries
```

`dcac37c1` is required for eval-only runs. In eval-only mode the rollout worker
does not construct a weight syncer; it loads directly from
`rollout.model.model_path`. Calling actor-to-rollout synchronization would crash,
and guarding only the rollout receiver would instead leave the actor sender
waiting forever.

`18f0c0e4` propagates MetaWorld task IDs as training metadata and aggregates
router sufficient statistics by task across micro-batches, update epochs, and
distributed ranks. It intentionally removes task IDs before calling OpenPI, so
they do not alter the policy input or routing function.

`7cac562f` keeps the detailed per-sample router tensors only on models configured
with task-conditioned logging. Default GSE and rollout workers retain only the
original aggregate diagnostics, avoiding unnecessary memory growth, especially
for token routing.

`d292f580` adds matched-throughput rollout profiles for one and two Pi0.5
rollout processes per GPU. It also validates that environment batches are
divisible by the rollout worker count before workers launch.

`8646d1a0` adds independent rollout-capacity and actor-activation profiles. The
recommended 8 x A100 80 GB composition is `metaworld_pi05_batch32` plus
`pi05_micro256`; the batch64 profile is an optional capacity probe, not the
default.

`c798a718` makes Flow-SDE rollout randomness reproducible with a base seed plus
rollout rank, writes complete scalar tables to `metrics.jsonl`, adds short aliases
for otherwise truncated task-router metrics, and provides a matched multi-seed
summary tool.

## 5. Validation completed

### 5.1 Model and gradient validation

The real MetaWorld SFT checkpoint produced:

```text
total parameters:                         3,646,909,201
injected GSE layers:                      126
GSE adapter parameters:                  28,938,240
trainable parameters including value:    30,151,681
```

Verified properties:

- every GSE B matrix starts at zero;
- maximum BF16 A-orthogonality error was about `6.856e-4`;
- initial base-versus-GSE output difference was exactly `0.0`;
- real CUDA policy forward returned finite actions, log-probabilities, and values;
- the first backward trains B, and subsequent updates give A/router gradients;
- two real FSDP PPO recomputations plus AdamW updates succeeded;
- weight-sync parameter names were unique.

The first-update B-only behavior is expected from zero-output initialization and
is not a router failure.

### 5.2 8 x A100 smoke and resume

Completed successfully:

- two PPO steps on 8 x A100 80 GB;
- rollout, advantage computation, actor/critic updates, and weight sync;
- checkpoint save;
- resume from checkpoint and completion of the next step.

Observed resumed-run diagnostics included:

```text
active GSE layers:          126
load-balance diagnostic:   1.2
orthogonality diagnostic:  4.88e-9
normalized router entropy: 0.966
expert selection range:    0.132 - 0.188
```

### 5.3 Short multi-task pilots

The first unbalanced 10-step run is retained only as an infrastructure result,
because task sampling was not controlled:

```text
eval success_once:   58.203%
success_at_end:      39.0625%
macro success:       58.1%
tasks >= 90%:        17
worst-10 mean:       6.6%
```

The balanced 10-step run used `actor.optim.total_training_steps=1000`, so the
actor learning rate remained at `5e-5` instead of decaying over ten steps:

```text
eval success_once:        55.078125%
success_at_end:           39.257812%
macro success:            54.9%
tasks >= 90%:             15
worst-10 mean:            2.9%
zero-success task IDs:    12, 19, 25, 28, 30, 40, 47
normalized router entropy: 0.967
expert selection range:   0.145 - 0.190
expert probability range: 0.161 - 0.175
load-balance diagnostic:  1.121
orthogonality diagnostic: 9.43e-7
```

All 50 tasks appeared in training. The run took about 34 minutes 30 seconds for
ten steps; the final step including evaluation took about 487 seconds.

### 5.4 Paired SFT versus 10-step GSE evaluation

Both policies were evaluated through the corrected eval-only runner with 512
trajectories, all 50 tasks, fixed reset-state IDs, and the same model/input
settings:

| Metric           |       SFT | GSE step 10 |       Delta |
| ---------------- | --------: | ----------: | ----------: |
| `success_once`   |   46.875% |   56.83594% | +9.96094 pp |
| `success_at_end` | 34.17969% |   39.25781% | +5.07812 pp |
| task macro mean  |     46.7% |       56.6% |     +9.9 pp |
| tasks above 90%  |        13 |          17 |          +4 |
| worst-10 mean    |      2.8% |        1.8% |     -1.0 pp |
| worst-5 mean     |      0.0% |        0.0% |      0.0 pp |

At the trajectory level, GSE adds 51 `success_once` successes (`240 -> 291`) and
26 `success_at_end` successes (`175 -> 201`). At the per-task level, 21 tasks
improve, 10 regress, and 19 are unchanged at the available 10/11 trials per task.

GSE unlocks three SFT-zero tasks (`02`, `11`, and `22`), while four previously
nonzero tasks become zero (`12`, `19`, `28`, and `47`). The largest gains include
task `11` (+100 pp), `37` (+70 pp), `14` and `21` (+50 pp). The largest
regressions include task `09` (-45.4 pp), `19` and `47` (-40 pp), `12` and `46`
(-30 pp).

Interpretation:

- The aggregate short-run RL benefit is real under the matched evaluation
  pipeline; the previous external 43.8% number is no longer needed for this
  comparison.
- The method has not yet balanced MT50: the worst-10 tail regresses slightly and
  the number of zero-success tasks changes from seven to eight.
- Per-task estimates have high variance with only 10/11 trials. Repeat both SFT
  and GSE evaluation over multiple matched random seeds before treating individual
  task deltas as stable.
- Global router entropy remains evidence against collapse, but cannot explain
  which tasks gained or regressed. Task-conditioned router statistics are the
  next implementation milestone.

### 5.5 Validated Profile A 10-step result

The user-validated faster Profile A used 128 environments, `rollout_epoch=2`,
`actor.micro_batch_size=128`, and `actor.global_batch_size=1024`. Its step-10
evaluation used 512 fixed-reset trajectories:

| Metric           |      SFT | GSE step 10 |       Delta |
| ---------------- | -------: | ----------: | ----------: |
| `success_once`   | 46.8750% |    57.6172% | +10.7422 pp |
| `success_at_end` | 34.1797% |    39.6484% |  +5.4687 pp |
| task macro mean  |   46.70% |      57.50% |    +10.8 pp |
| tasks above 90%  |       13 |          16 |          +3 |
| worst-10 mean    |    2.80% |       5.70% |     +2.9 pp |
| worst-5 mean     |    0.00% |       0.00% |      0.0 pp |

Using the displayed rounded per-task values, 25 tasks improve, nine regress, and
16 are unchanged. Zero-success tasks fall from seven to five (`01`, `26`, `28`,
`40`, and `42`). PPO remains in a healthy short-run regime: approximate KL is
`0.0017`, clip fraction is `0.014`, critic explained variance is `0.617`, and
normalized router entropy is `0.967`.

This run supersedes the older Profile A/B throughput decision, but it does not
supersede the need for matched seeds. Per-task estimates still have high sampling
variance, and the exact task-router NMI/JS values were truncated in the terminal.
Future runs persist those values in `metrics.jsonl`.

### 5.6 High-memory capacity result

The 8 x A100 80 GB capacity probe established:

- `metaworld_pi05_batch32 + pi05_micro256` runs without OOM;
- `metaworld_pi05_batch64` exceeds available memory;
- the first batch32/micro256 global step took `497.8 s`, including `261.0 s`
  rollout generation and `231.5 s` actor training;
- the first batch16/epoch4 global step was even slower at `546.3 s`, including
  `311.1 s` rollout generation and `229.9 s` actor training.

This is a memory-capacity success but not a throughput success. Compared with
the validated batch16 run's approximately `207.1 s` non-eval step, five batch32
steps project to about 41.5 minutes, versus about 34.5 minutes for ten batch16
steps at the same 2,560-trajectory budget. The comparison uses a batch32 first
step and a mature batch16 step, so record the batch32 second step as well, but do
not select the configuration merely because it occupies more GPU memory.

The likely bottlenecks are increased concurrent environments, repeated bootstrap
overhead, and the actor micro batch of 256. Per collected sample, both rollout
and actor time degraded; the larger actor micro batch did not improve kernel
throughput. The formal run therefore returns to the validated batch16 settings
using command-line overrides. Do not add more throughput YAML profiles.

## 6. Eval-only behavior and reproducible evaluation

### 6.1 Critical loading rule

With `runner.only_eval=True`:

- rollout weights come only from `rollout.model.model_path`;
- actor weights are not synchronized to rollout;
- `runner.resume_dir` must not be used as a substitute for the rollout path;
- actor model construction may still require a valid SFT model path, but it does
  not determine the evaluated rollout policy.

The rollout model group must be explicitly composed with:

```text
'+model@rollout.model=pi0_5'
```

### 6.2 Raw SFT balanced evaluation

Inside Docker:

```bash
export RUN_DIR=/workspace/output/sft-balanced-eval
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse \
  '+model@rollout.model=pi0_5' \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.num_action_chunks=5 \
  rollout.model.action_dim=4 \
  rollout.model.openpi.config_name=pi05_metaworld \
  rollout.model.openpi.num_images_in_input=1 \
  runner.only_eval=True \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name=sft_balanced_eval \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  2>&1 | tee "$RUN_DIR/console.log"
```

### 6.3 GSE checkpoint balanced evaluation

Each RL checkpoint contains distributed training state and a full model weight
file. Point the rollout model at the checkpoint's `actor` directory containing
`model_state_dict/full_weights.pt`; do not rely on `resume_dir`.

Example:

```bash
export GSE_CKPT=/workspace/output/gse-balanced-observe-seed42/gse_balanced_observe_seed42/checkpoints/global_step_10
export RUN_DIR=/workspace/output/gse-balanced-observe-seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse \
  '+model@rollout.model=pi0_5' \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path="$GSE_CKPT/actor" \
  rollout.model.num_action_chunks=5 \
  rollout.model.action_dim=4 \
  rollout.model.openpi.config_name=pi05_metaworld \
  rollout.model.openpi.num_images_in_input=1 \
  rollout.model.gse.enabled=True \
  rollout.model.gse.total_rank=64 \
  rollout.model.gse.num_experts=8 \
  rollout.model.gse.num_generalized_experts=2 \
  rollout.model.gse.top_k=2 \
  '+rollout.model.openpi_data.norm_stats_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT/lerobot/metaworld_mt50/norm_stats.json' \
  runner.only_eval=True \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name=gse_balanced_eval \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  2>&1 | tee "$RUN_DIR/console.log"
```

Before launching, verify the checkpoint layout:

```bash
test -f "$GSE_CKPT/actor/model_state_dict/full_weights.pt"
```

Use the exact GSE architecture from the training run if it differs from the
defaults above. Both evaluations must use the same reset-state configuration and
trial count. Compare at least:

- `success_once` and `success_at_end` separately;
- macro task success, median, worst-5, and worst-10;
- number of tasks above 90%;
- per-task SFT-to-GSE deltas and number of regressed tasks;
- zero-success tasks that become nonzero.

## 7. Server and Docker runbook

The following real host paths have already been selected. Preserve them exactly:

```bash
export RLINF_REPO=/home/xueyang/RLinf
export PI05_SFT=/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-SFT
export GSE_OUTPUT=/DATA/disk0/model/pi05-gse
export HF_CACHE=/home/xueyang/RLinf/cache/huggingface
export RAY_SCRATCH=/DATA/disk0/xueyang/Data/rlinf-ray
mkdir -p \
  "$GSE_OUTPUT" \
  "$HF_CACHE" \
  "$RAY_SCRATCH/session" \
  "$RAY_SCRATCH/spill" \
  "$RAY_SCRATCH/tmp"
```

Start Docker:

```bash
docker run -it --rm \
  --gpus all \
  --shm-size 256g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name rlinf-pi05-gse \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e RLINF_RAY_TEMP_DIR=/workspace/ray/session \
  -e RLINF_RAY_OBJECT_SPILL_DIR=/workspace/ray/spill \
  -e TMPDIR=/workspace/ray/tmp \
  -v "$RLINF_REPO":/workspace/RLinf \
  -v "$PI05_SFT":/workspace/models/RLinf-Pi05-MetaWorld-SFT:ro \
  -v "$GSE_OUTPUT":/workspace/output \
  -v "$RAY_SCRATCH":/workspace/ray \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /workspace/RLinf \
  rlinf/rlinf:agentic-rlinf0.3-maniskill_libero \
  bash
```

Initialize the container:

```bash
source switch_env openpi
cd /workspace/RLinf

export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### Docker space issue

Ray previously wrote hundreds of GB under `/tmp/ray`, filling Docker's overlay
filesystem. The current code and container setup redirect session state, object
spill, and temporary files to the host-backed `$RAY_SCRATCH` mount:

```text
RLINF_RAY_TEMP_DIR=/workspace/ray/session
RLINF_RAY_OBJECT_SPILL_DIR=/workspace/ray/spill
TMPDIR=/workspace/ray/tmp
```

The GSE config also disables unnecessary video output, and checkpoint retention
is bounded to the latest two checkpoints. Monitor the host-backed directory with:

```bash
du -sh "$RAY_SCRATCH"/*
df -h "$RAY_SCRATCH"
docker system df
```

Do not run broad Docker prune commands on a shared server. Remove only confirmed
stale Ray session directories after all associated jobs and containers have
stopped.

## 8. Tests

The eval-only regression test is:

```bash
python -m pytest -q tests/unit_tests/test_embodied_eval_only.py
```

It passed in the OpenPI Docker environment after `dcac37c1`.

Before a long run, execute the focused suite:

```bash
python -m pytest -q \
  tests/unit_tests/test_embodied_eval_only.py \
  tests/unit_tests/test_metaworld_reset_sampling.py \
  tests/unit_tests/test_checkpoint_retention.py \
  tests/unit_tests/test_ray_storage_config.py \
  tests/unit_tests/test_task_success_metrics.py \
  tests/unit_tests/test_comm_mapper.py \
  tests/unit_tests/models/test_gse.py \
  tests/unit_tests/models/test_openpi_gse.py
```

This focused suite passed on 2026-07-17: `45 passed`. The placement and channel
routing tests for parallel rollout separately passed: `47 passed`. Ruff passed
on all changed Python files. Both the training config with task-conditioned
router metrics and the GSE eval-only overrides in Section 6.3 resolved
successfully in the same Docker image.

The 2026-07-17 high-memory and reproducibility changes passed `53` focused tests
plus Ruff. Hydra resolved the batch32, batch32-plus-micro256, and batch64
compositions in the OpenPI image. The final formal command-line overrides were
also resolved after the unsuccessful batch16-epoch4 profile was removed.

### Task-conditioned router metrics

The MetaWorld GSE config enables:

```yaml
actor:
  model:
    gse:
      log_task_router_metrics: true
      task_router_num_tasks: 50
```

The terminal table suppresses the detailed per-task/per-expert values to remain
readable, while TensorBoard receives them. Summary metrics remain visible under
`train/gse/task_router/`:

- `covered_tasks`: tasks present in PPO recomputation;
- `normalized_mutual_information`: dependence between task and selected expert;
- `mean_js_divergence`: average task distribution distance from global routing;
- `mean_selection_std_across_tasks`: variability of hard selection by task;
- `mean_probability_std_across_tasks`: variability of soft probabilities by task.

Short aliases `nmi`, `js`, `select_std`, and `prob_std` remain readable in the
terminal. The full names and detailed per-task values are written to
`metrics.jsonl` and TensorBoard.

Near-zero values mean the router behaves similarly across tasks. Larger values
show task dependence, but not necessarily useful specialization; correlate them
with per-task success deltas. Detailed keys have the form
`train/gse/task_router/task_XX/expert_Y_selection` and `_probability`.

Also resolve the Hydra config before expensive jobs:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse \
  --cfg job --resolve \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  > /tmp/pi05_gse_resolved.yaml
```

## 9. Formal one-day training

### 9.1 Final throughput decision

The fastest validated setting remains eight rollout replicas with batch 16 each:

```text
128 environments x rollout_epoch 2 = 256 trajectories/global step
256 trajectories x (100 / 5) = 5,120 PPO chunk samples/global step
actor global batch 1,024, update_epoch 4 = 20 optimizer updates/global step
```

Increasing memory occupancy did not improve end-to-end throughput. Batch32 with
micro256 took `497.8 s`, batch16 with four rollout epochs took `546.3 s`, and
batch64 OOMed. Use the validated batch16 values directly on the command line; do
not compose `pi05_micro256` and do not add another throughput YAML profile.

A 320-step run collects 81,920 trajectories, approximately 8.19 million
environment steps, and performs 6,400 optimizer updates. At the measured steady
step time, plus eighteen 512-trajectory evaluations, it should fit in roughly
19-23 hours. The cosine scheduler advances per optimizer update, not per global
step, so every launch and resume must retain
`actor.optim.total_training_steps=6400`.

### 9.2 Stage 1: train to step 40

Use command-line overrides for all run-specific values:

```bash
export RUN_DIR=/workspace/output/gse-formal-seed42
export EXP_NAME=gse_formal_seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  rollout.pipeline_stage_num=1 \
  actor.micro_batch_size=128 \
  actor.global_batch_size=1024 \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=40 \
  runner.save_interval=20 \
  runner.val_check_interval=10 \
  runner.max_checkpoints_to_keep=4 \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  actor.optim.lr=5e-5 \
  actor.optim.total_training_steps=6400 \
  actor.seed=42 \
  rollout.seed=42 \
  2>&1 | tee "$RUN_DIR/console-stage1.log"
```

This evaluates at steps 10, 20, 30, and 40, and saves at steps 20 and 40. Proceed
to Stage 2 only when step 40 has no hard failure signal from Section 9.5. A
temporarily flat success curve is not sufficient reason to stop because
sparse-reward MT50 evaluations are noisy.

### 9.3 Stage 2 and Stage 3 resume

Resume to step 120 from the step-40 checkpoint:

```bash
export CKPT="$RUN_DIR/$EXP_NAME/checkpoints/global_step_40"
test -d "$CKPT/actor"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  rollout.pipeline_stage_num=1 \
  actor.micro_batch_size=128 \
  actor.global_batch_size=1024 \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  runner.resume_dir="$CKPT" \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=120 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  runner.max_checkpoints_to_keep=4 \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  actor.optim.lr=5e-5 \
  actor.optim.total_training_steps=6400 \
  actor.seed=42 \
  rollout.seed=42 \
  2>&1 | tee "$RUN_DIR/console-stage2.log"
```

If the step-120 checkpoint remains healthy, run the same command to step 320 by
changing only:

```bash
export CKPT="$RUN_DIR/$EXP_NAME/checkpoints/global_step_120"
# In the Stage 2 command, use runner.resume_dir="$CKPT", runner.max_epochs=320,
# and tee "$RUN_DIR/console-stage3.log". Keep every other override unchanged.
```

Checkpoint resume restores model, optimizer, and LR scheduler. Do not change
`total_training_steps`, GSE architecture, optimizer, batch sizes, or seed between
stages. With four retained checkpoints, archive any checkpoint selected as a
candidate before later saves prune it.

### 9.4 Live monitoring

TensorBoard:

```bash
tensorboard --logdir "$RUN_DIR/tensorboard" --host 0.0.0.0 --port 6006
```

Compact exact metrics without relying on truncated terminal columns:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/workspace/output/gse-formal-seed42/metrics.jsonl")
keys = (
    "eval/success_once",
    "eval/success_at_end",
    "eval/task_success/macro_mean",
    "eval/task_success/worst_10_mean",
    "eval/task_success/num_above_90",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/grad_norm",
    "train/critic/explained_variance",
    "train/gse/router/normalized_entropy",
    "train/gse/task_router/nmi",
    "train/gse/task_router/js",
)
for line in path.open():
    record = json.loads(line)
    metrics = record["metrics"]
    if "eval/success_once" in metrics:
        checkpoint_step = record["step"] + 1
        print(checkpoint_step, {key: metrics.get(key) for key in keys})
PY
```

Training records use a zero-based loop index, hence the `+1` above; standalone
eval-only records do not need this conversion.

Also monitor `time/step`, all eight GPU memories, host RAM, Ray scratch usage,
and remaining disk space. Training success metrics (`env/success_once`) are useful
for trend detection but are not a convergence criterion because policy and reset
samples change continuously.

### 9.5 Convergence and stopping rules

Select checkpoints by fixed-reset evaluation, not actor loss. A practical
convergence judgment requires all of the following:

- `eval/success_once`, macro mean, and worst-10 improve or remain stable across
  at least three evaluations (40 global steps); use a moving three-evaluation
  average rather than one point;
- `success_at_end` tracks `success_once`; a widening gap can indicate transient
  task completion that the policy fails to maintain;
- the number of tasks above 90% rises without a growing count of zero-success or
  regressed tasks relative to the SFT baseline;
- approximate KL and clip fraction remain finite and controlled, critic explained
  variance does not persistently collapse below zero, and gradients remain finite;
- router entropy does not collapse toward zero, all 50 tasks remain covered, and
  task-router NMI/JS are interpreted together with per-task success rather than as
  objectives by themselves.

Stop and roll back to the best checkpoint if two consecutive evaluations drop
macro success by more than 5 percentage points from the best checkpoint, worst-10
degrades materially, approximate KL repeatedly exceeds `0.02`, clip fraction
repeatedly exceeds `0.2`, gradients become non-finite, or router entropy collapses.
Treat convergence as reached when the three-evaluation moving averages of macro
success and worst-10 improve by less than 1 percentage point while no tail task
regresses. Confirm the selected checkpoint with three matched rollout seeds; a
single 512-trajectory evaluation is not a paper result.

### 9.6 Multi-seed evaluation

New runs append every scalar metric, including untruncated router NMI/JS/std, to
`$RUN_DIR/metrics.jsonl`. For final comparisons, evaluate both SFT and GSE with
the same rollout seeds, for example `42`, `43`, and `44`. Add
`rollout.seed=$SEED` to the Section 6 commands and use a separate `RUN_DIR` for
each seed. Fixed reset-state IDs keep environment initial states matched while
the seed controls stochastic Flow-SDE sampling.

Summarize the resulting files with:

```bash
python -m toolkits.embodiment.summarize_multiseed_eval \
  /workspace/output/gse-eval-seed*/metrics.jsonl \
  --baseline /workspace/output/sft-eval-seed*/metrics.jsonl \
  --output /workspace/output/gse-vs-sft-multiseed.json
```

The summary reports mean, sample standard deviation, approximate 95% confidence
intervals, per-task mean deltas, and improved/regressed/unchanged task counts.

### 9.7 Improvements after the baseline

Run the 320-step configuration unchanged first. Then test one change at a time:

1. Add task-wise advantage normalization because global normalization can let
   easy/high-variance tasks dominate multi-task PPO gradients.
2. Use adaptive task sampling or capped tail-task weighting based on a smoothed
   per-task success estimate; cap weights to avoid overfitting zero-success tasks.
3. Add a small SFT behavior-cloning anchor or reference-policy KL only if long-run
   evaluation shows forgetting. Both cost compute/memory and are not justified by
   the current positive transfer result.
4. Tune router temperature/top-k or a very small load-balance coefficient only
   after NMI/JS plus per-task deltas show collapse or harmful specialization.
   Orthogonality is already near zero and does not currently need a penalty.
5. Compare GSE against full-parameter PPO and parameter-matched LoRA PPO using
   identical trajectories, seeds, reset states, optimizer updates, and evaluation.
6. After selecting the objective, run at least three training seeds and held-out
   visual/state perturbations before claiming improved generalization.

### 9.8 Current method summary

The current policy starts from the fully trained Pi0.5 MetaWorld MT50 SFT
checkpoint. The VLM and original action-expert parameters stay frozen. GSE wraps
126 linear projections in the 18-layer action expert with two always-active
generalized experts and six routed specialized experts (`top_k=2`, total rank
64). Orthogonal A matrices and zero B matrices make the initial residual exactly
zero, so the initial policy is identical to SFT.

RLinf then uses Flow-SDE/Flow-Noise PPO to collect balanced MT50 trajectories.
Only GSE/router parameters and the value head are optimized. Each global step
collects 256 trajectories, computes GAE with a shared multi-task value head, and
runs four PPO update epochs over five global minibatches. Load-balance and
orthogonality losses remain disabled; router/task specialization is monitored
rather than forced. The formal objective is therefore parameter-efficient
residual multi-task RL post-training of a frozen SFT VLA, with fixed-reset
per-task evaluation used to detect positive transfer, conflict, and forgetting.

For paper results, report mean, median, worst-5/worst-10, tasks above 90%, negative
transfer relative to SFT, environment steps, trainable parameters, optimizer
memory, peak GPU memory, and wall time. Include held-out visual/state perturbation
tests if making a generalization claim.

## 10. Development cautions

- Read the repository-root `AGENTS.md` before editing.
- Preserve unrelated user changes and real server paths.
- Keep the official full-parameter Pi0.5 PPO config unchanged as a baseline.
- Do not expand GSE into the VLM backbone before the action-expert experiment is
  stable and an ablation justifies it.
- Zero-B initialization delays useful A/router gradients until B becomes nonzero.
- Aggregate load balance over a diverse multi-task batch; per-sample balancing can
  be meaningless or harmful.
- Do not claim specialization from high entropy or uniform aggregate usage alone.
- Do not terminate unrelated GPU jobs or delete shared Docker/Ray data.
- Put checkpoints, TensorBoard data, videos, and large logs outside the repository.

The immediate milestone is Stage 1 of the formal batch16 run, followed by the
step-40 convergence review. Do not enable auxiliary losses until the baseline
run and matched evaluations show whether gains and regressions correspond to
useful expert specialization.
