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

The main research claim is not established yet. The first balanced 10-step run is
a positive signal, but its SFT comparison was not produced by the same eval-only
pipeline. A paired evaluation is the next required experiment.

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
does not prove task specialization. Task-conditioned expert utilization is still
missing and is the most important observability feature to add next.

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
```

`dcac37c1` is required for eval-only runs. In eval-only mode the rollout worker
does not construct a weight syncer; it loads directly from
`rollout.model.model_path`. Calling actor-to-rollout synchronization would crash,
and guarding only the rollout receiver would instead leave the actor sender
waiting forever.

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

### 5.4 Interpretation of the reported 43.8% SFT score

Do not yet claim an improvement from `43.8%` to `55.1%`.

That conclusion is valid only if 43.8% used the same checkpoint, observation and
action settings, fixed reset states, number of trials per task, and specifically
the same `success_once` metric. If 43.8% is `success_at_end`, the comparable GSE
number is 39.26%, not 55.08%.

The balanced pilot is therefore a promising signal, not causal evidence. Its low
worst-10 score can primarily reflect the weak/uneven SFT initialization, sparse
reward, and only ten PPO updates. The paired eval described below must be run
before changing the method or starting a long experiment.

## 6. Eval-only behavior and paired evaluation

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
export GSE_CKPT=/workspace/output/<run>/<global_step_checkpoint>
export RUN_DIR=/workspace/output/gse-balanced-eval
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
export GSE_OUTPUT=/home/xueyang/RLinf/pi05-gse
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
  tests/unit_tests/models/test_gse.py \
  tests/unit_tests/models/test_openpi_gse.py
```

This focused suite passed on 2026-07-17: `36 passed`. The GSE eval-only Hydra
overrides in Section 6.3 were also resolved successfully in the same Docker image.

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

## 9. Next work, in order

1. Pull `dcac37c1` onto `/home/xueyang/RLinf` and run the eval-only unit test.
2. Run the raw SFT balanced evaluation from Section 6.2.
3. Run the selected balanced GSE checkpoint evaluation from Section 6.3 using
   the identical reset states and 512 trials.
4. Produce a per-task delta table. Decide whether the pilot improves average,
   tails, or only a subset of already-easy tasks.
5. Add task-conditioned router/expert utilization. Aggregate selection and
   probability by MT50 task ID, not only globally.
6. If task imbalance remains in PPO batches, add task-wise advantage
   normalization or controlled worst-task weighting. Do not add it before
   measuring task frequencies and per-task gradient/advantage scales.
7. Run matched short pilots for GSE PPO, full-parameter PPO, and parameter-matched
   plain LoRA PPO. Match environment steps, reset states, seeds, and evaluation
   frequency.
8. Only after these checks, run multi-seed longer training and tune auxiliary
   losses.

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

The immediate milestone is a paired, fixed-reset SFT-versus-GSE evaluation. Do not
start a long experiment or enable auxiliary losses until that comparison is
available.
