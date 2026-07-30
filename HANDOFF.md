# Pi0.5 Multi-Task Parameter-Efficient RL Handoff

Last updated: 2026-07-30

This document is the complete handoff for the Pi0.5 MetaWorld MT50 project. It
retains validated experiments, output/checkpoint paths, reproducible commands,
implementation constraints, and current paper conclusions. Read it before
changing code, evaluating a checkpoint, or launching a long run.

## 1. Research objective and current hypothesis

The goal is to improve one MetaWorld MT50 VLA policy on many tasks at once with
parameter-efficient RL post-training, while reducing multi-task interference and
retaining the SFT policy as an exact initialization. The RL algorithm and critic
are held fixed to RLinf's standard Flow-SDE PPO/GAE setup; the research variable
is the trainable policy parameterization.

The current method is:

1. Load RLinf's fully supervised Pi0.5 MetaWorld MT50 checkpoint.
2. Insert zero-output GSE residual adapters into the Pi0.5 action expert.
3. Freeze the pretrained base policy.
4. Use RLinf's Flow-SDE PPO path to update GSE, its router, and the
   value head.
5. Compare against the raw SFT policy, official full-parameter PPO, and a
   parameter-matched plain LoRA PPO baseline under matched data and evaluation
   budgets.

This is not a second GSE-SFT stage and does not decompose an SFT weight delta.
Orthogonal A initialization plus zero B initialization makes the initial GSE
policy exactly equal to the loaded SFT policy. RL learns only a residual.

The selected action-GSE checkpoint is step 180, the highest checkpoint in the
converged step-20-to-step-220 region. Three stochastic rollout seeds from this
one training run average `72.72%` success-once and `72.73%` task macro success.
The central current result is therefore `+2.02 pp` over RLinf's reported `70.7%`
average while training 30.15M parameters of a 3.65B-parameter policy.

This is not yet evidence of statistically significant superiority over the
released full-parameter checkpoint. A local one-seed, 512-trajectory evaluation
of that checkpoint gives `74.22%`; it still needs the same three rollout seeds
and reset protocol. The present paper claim is parameter-efficient multi-task RL
post-training that exceeds the reported RLinf number, not a completed matched
proof that it beats full-parameter RL.

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

Server model and result roots are recorded in Section 7. In particular, preserve
the released RL checkpoint at
`/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-RL-FlowSDE` and all experiment
outputs under `/DATA/disk0/xueyang/model/pi05-gse`.

### 2.1 Training and evaluation output index

`/workspace/output` is the Docker mount of
`/DATA/disk0/xueyang/model/pi05-gse`. Preserve these paths:

| Artifact | Docker path |
|---|---|
| Action-GSE formal run | `/workspace/output/gse-formal-seed42` |
| Action-GSE metrics | `/workspace/output/gse-formal-seed42/metrics.jsonl` |
| Selected action-GSE checkpoint | `/workspace/output/gse-formal-seed42/gse_formal_seed42/checkpoints/global_step_180` |
| Archived step-180 copy | `/workspace/output/gse-formal-seed42/best_global_step_180` |
| Action-GSE three-seed evaluation | `/workspace/output/gse-step180-multiseed` |
| Action-GSE three-seed summary | `/workspace/output/gse-step180-multiseed/summary.json` |
| Raw SFT balanced evaluation | `/workspace/output/sft-balanced-eval` |
| Released full-RL seed-42 evaluation | `/workspace/output/rlinf-flow-sde-eval-seed42` |
| Plain-LoRA training | `/workspace/output/action-lora-r64-seed42` |
| Plain-LoRA selected checkpoint | `/workspace/output/action-lora-r64-seed42/action_lora_r64_seed42/checkpoints/global_step_80` |
| Plain-LoRA step-80 evaluation | `/workspace/output/action-lora-r64-step80-eval-seed42` |
| VLM last-block run | `/workspace/output/gse-action180-vlm-last-seed42` |
| VLM last-four run | `/workspace/output/gse-action180-vlm-last4-seed42` |
| Action-GSE rank-32 run | `/workspace/output/gse-action-r32-seed42` |
| Joint action + VLM-last4 from-SFT run | `/workspace/output/gse-joint-vlm-last4-seed42` |
| Joint action + all-VLM-layers from-SFT run | `/workspace/output/gse-joint-vlm-all-seed42` |

For any path above, replace the `/workspace/output` prefix with
`/DATA/disk0/xueyang/model/pi05-gse` to obtain its host path. Checkpoints and
raw `metrics.jsonl` files are the source of truth; this document summarizes them
but does not replace them.

Docker image:

```text
rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
```

Activate the OpenPI environment inside Docker with:

```bash
source switch_env openpi
```

The target server has 8 x A100 80 GB.

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

The main action-GSE result injects GSE only into the 18-layer Pi0.5 action
expert. The seven wrapped projections per layer are `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, for 126 adapters in
total. The optional VLM-last-four extension is documented in Section 9.11.

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

The selected action-GSE run used monitoring-only auxiliary losses. Keep
`load_balancing_loss_coef=0` and `orthogonality_loss_coef=0` until paired
evaluation and task-conditioned router measurements justify changing the
objective.

Uniform aggregate expert usage is evidence against global router collapse, but it
does not prove task specialization. Task-conditioned expert utilization is now
logged during PPO recomputation and should be used for that judgment.

### 3.4 Action-head plain LoRA baseline

The matched action-head baseline is implemented in:

```text
examples/embodiment/config/metaworld_50_ppo_openpi_pi05_action_lora.yaml
rlinf/models/embodiment/openpi/lora.py
tests/unit_tests/models/test_openpi_action_lora.py
```

It uses rank 64, alpha 64, PEFT Gaussian A and zero B, and wraps the same seven
projection types across all 18 action-transformer layers as GSE (126 projections
total). The pretrained policy is frozen; only action LoRA and the value head are
trained. This dedicated implementation is required because RLinf's older generic
OpenPI LoRA path targets the VLM rather than the action expert.

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
c798a718  reproducible rollout seeds and multi-seed evaluation summaries
8c59a4a9  VLM GSE upgrade from action-only checkpoints
51fe065d  action-head plain-LoRA baseline and full-checkpoint loading
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

`c798a718` makes Flow-SDE rollout randomness reproducible with a base seed plus
rollout rank, writes complete scalar tables to `metrics.jsonl`, adds short aliases
for otherwise truncated task-router metrics, and provides a matched multi-seed
summary tool.

`8c59a4a9` upgrades action-only checkpoints by loading trained action GSE before
injecting optional zero-output VLM GSE. It supports freezing action adapters,
selected VLM layer indices, domain-separated router metrics, and safe loading of
future joint checkpoints.

## 5. Validation completed

Current aggregate comparison (protocol differences are explicit and must not be
hidden in the paper):

| Method | Evaluation protocol | Success once | Macro | End | Worst-10 | >=90 tasks |
|---|---|---:|---:|---:|---:|---:|
| RLinf reported Flow-SDE | reported average | 70.70% | unavailable | unavailable | unavailable | unavailable |
| Released full Flow-SDE | 1 rollout seed, 512 trajectories | 74.22% | 74.20% | 42.58% | 16.20% | 18 |
| Action-GSE step 180 | 3 rollout seeds, 512 trajectories each | 72.72% | 72.73% | 47.20% | 18.73% | 21.0 |
| Plain LoRA step 80 | 1 rollout seed, 448 trajectories | 63.17% | 63.22% | 42.86% | 4.44% | 19 |

The headline comparison is action-GSE `72.72%` versus the reported `70.7%`.
The released full checkpoint's local `74.22%` result is a separate, currently
unmatched comparison; do not replace the reported baseline with it silently or
claim GSE beats it until the protocol is matched.

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

### 5.3 Formal action-GSE run through step 220

The first eleven fixed-reset evaluations of the formal batch16 run are:

| Checkpoint | `success_once` | Macro mean |   Worst-10 | Tasks >=90% |
| ---------: | -------------: | ---------: | ---------: | ----------: |
|         20 |         65.63% |     65.36% |      9.64% |          20 |
|         40 |         66.99% |     66.95% |     12.64% |          21 |
|         60 |         70.90% |     70.78% |      9.73% |          24 |
|         80 |         68.16% |     68.13% |     15.82% |          18 |
|        100 |         67.58% |     67.55% |      9.64% |          20 |
|        120 |         69.14% |     69.09% |     10.45% |          22 |
|        140 |         69.73% |     69.62% |     12.64% |          18 |
|        160 |         67.58% |     67.67% |      9.64% |          20 |
|        180 |     **73.05%** | **73.13%** | **18.18%** |          21 |
|        200 |         67.58% |     67.51% |     13.82% |          19 |
|        220 |         68.95% |     68.82% |     12.55% |          19 |

The step-180 checkpoint is currently the best observed checkpoint. It exceeds
the reported official Flow-SDE `70.7%` by `2.35 pp` on `success_once` and by
`2.43 pp` on task macro mean. This is a promising provisional result, not yet a
strict comparison: both methods must use the same success field, 512-trajectory
count, task reset pool, checkpoint selection rule, and rollout seed protocol.
The step-200 regression also shows that one evaluation is too noisy to support a
final claim. The run is treated as converged over the observed step-20-to-step-220
window; keep `global_step_180` as the selected candidate and do not resume it
solely to search for a higher single-seed checkpoint.

The router remains non-collapsed but effectively task-agnostic: at step 180,
normalized entropy is `0.954`, task-router NMI is `5.63e-4`, and JS divergence is
`2.53e-4`. Do not claim that the current gain comes from task specialization;
the current evidence supports a shared residual improvement across tasks.

The matched three-seed evaluation of step 180 gives:

```text
success_once: mean 72.72%, std 2.16%, approximate 95% CI [70.28%, 75.17%]
macro mean:   mean 72.73%, std 2.07%, approximate 95% CI [70.38%, 75.07%]
success_end:  mean 47.20%, std 1.24%
worst-10:     mean 18.73%, std 6.32%
worst-5:      mean  6.91%, std 6.50%
```

Seed 42 is the highest run. The three-seed mean is `2.02 pp` above the reported
official 70.7%, but 70.7% remains inside the approximate confidence interval.
The result supports moving to an isolated VLM-GSE ablation, while a strict
superiority claim still requires matched official checkpoints/seeds.

These are three stochastic evaluation seeds from the same seed-42 training
checkpoint, not three independent training runs. The confidence intervals only
measure rollout variation. Mean tasks above 90% is `21.0`.

### 5.4 Released full Flow-SDE checkpoint evaluation

The released full-parameter model is stored on the server at:

```text
/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-RL-FlowSDE
/workspace/models/RLinf-Pi05-MetaWorld-RL-FlowSDE  # inside Docker
```

One eight-GPU evaluation with 512 trajectories produced:

| Metric | Released full Flow-SDE |
|---|---:|
| `success_once` | 74.21875% |
| `success_at_end` | 42.578125% |
| task macro mean | 74.20% |
| worst-10 mean | 16.20% |
| worst-5 mean | 2.00% |
| tasks above 90% | 18 |

Per-task success for task IDs 00-49 was:

```text
00-09: 0.727, 0.000, 0.364, 0.455, 1.000, 1.000, 1.000, 1.000, 1.000, 0.818
10-19: 0.727, 0.818, 0.700, 1.000, 0.700, 0.900, 0.900, 0.800, 1.000, 0.700
20-29: 1.000, 1.000, 0.700, 1.000, 1.000, 0.000, 0.000, 0.700, 0.800, 0.700
30-39: 0.000, 0.800, 1.000, 0.900, 0.800, 0.900, 1.000, 1.000, 1.000, 0.200
40-49: 0.100, 0.800, 0.300, 0.900, 0.900, 0.200, 1.000, 0.800, 1.000, 1.000
```

This one rollout seed is higher than the action-GSE three-seed mean by about
`1.50 pp` on success-once. Action-GSE is better on success-at-end (`+4.62 pp`),
worst-10 (`+2.53 pp`), and tasks above 90% (`+3`), but those differences are not
formal until both checkpoints use identical seeds and reset states.

### 5.5 Seven-GPU action-head plain-LoRA run

The rank-64 plain-LoRA run used GPU ranks 1-7, 224 training trajectories per
global step, one rollout epoch, actor micro batch 128, global batch 896, and 448
fixed-reset evaluation trajectories. Its canonical output/checkpoint locations
are:

```text
/workspace/output/action-lora-r64-seed42
/workspace/output/action-lora-r64-seed42/action_lora_r64_seed42/checkpoints/global_step_<STEP>
```

The complete available evaluation curve is:

| Step | Success once | Success at end | Macro | Worst-10 | Tasks >=90% |
|---:|---:|---:|---:|---:|---:|
| 20 | 61.38% | 44.20% | 61.42% | 4.44% | 17 |
| 40 | 61.38% | 41.74% | 61.53% | 4.44% | 13 |
| 60 | 61.16% | 43.08% | 61.33% | 4.44% | 16 |
| 80 | **63.17%** | 42.86% | **63.22%** | 4.44% | 19 |
| 100 | 60.27% | 39.51% | 60.39% | 5.56% | 14 |
| 120 | 60.94% | 40.18% | 61.11% | 2.22% | 19 |
| 140 | 58.93% | 37.28% | 59.11% | 4.44% | 13 |
| 160 | 62.50% | 40.63% | 62.67% | **7.78%** | 17 |

Step 80 is the best macro checkpoint through step 160. PPO is numerically
healthy: approximate KL remains below `0.008`, clip fraction below `0.108`,
gradients are finite, and critic explained variance stays around `0.66-0.77`.
Router metrics are `None` by design because this baseline has no router.

Action-GSE exceeds the current best plain-LoRA macro result by `9.50 pp` and
worst-10 by `14.28 pp`. This strongly suggests that equal total low-rank capacity
alone does not explain the GSE result, but the final paper comparison still needs
matched trajectory counts, seeds, GPU placement, and RL environment-step budgets.

## 6. Eval-only behavior and reproducible evaluation

All commands in Sections 6 and 9 assume that the Docker shell has first been
initialized as described in Section 7, including the `WANDB_OVERRIDES` array.

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
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

### 6.3 GSE checkpoint balanced evaluation

Each RL checkpoint contains distributed training state and a full model weight
file. Point the rollout model at the checkpoint's `actor` directory containing
`model_state_dict/full_weights.pt`; do not rely on `resume_dir`.

Example:

```bash
export GSE_CKPT=/workspace/output/gse-formal-seed42/gse_formal_seed42/checkpoints/global_step_180
export RUN_DIR=/workspace/output/gse-step180-eval-seed42
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
  runner.logger.experiment_name=gse_step180_eval_seed42 \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  "${WANDB_OVERRIDES[@]}" \
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

### 6.4 Released full Flow-SDE evaluation

The following is the reproducible eight-GPU, 512-trajectory evaluation command.
It saves the complete metric table instead of leaving the result only in terminal
output:

```bash
export RL_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-RL-FlowSDE
export RUN_DIR=/workspace/output/rlinf-flow-sde-eval-seed42
test -f "$RL_MODEL/model.safetensors"
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05 \
  '+model@rollout.model=pi0_5' \
  actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  rollout.model.model_path="$RL_MODEL" \
  rollout.model.num_action_chunks=5 \
  rollout.model.action_dim=4 \
  rollout.model.openpi.config_name=pi05_metaworld \
  rollout.model.openpi.num_images_in_input=1 \
  rollout.model.openpi.noise_method=flow_sde \
  '+rollout.model.openpi_data.norm_stats_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT/lerobot/metaworld_mt50/norm_stats.json' \
  rollout.seed=42 \
  runner.only_eval=True \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name=rlinf_flow_sde_eval_seed42 \
  env.eval.total_num_envs=512 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Repeat with rollout seeds 43 and 44 and separate output directories before a
matched full-RL-versus-GSE claim.

## 7. Server and Docker runbook

The following real host paths have already been selected. Preserve them exactly:

```bash
export RLINF_REPO=/home/xueyang/RLinf
export PI05_SFT=/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-SFT
export PI05_RL=/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-RL-FlowSDE
export GSE_OUTPUT=/DATA/disk0/xueyang/model/pi05-gse
export HF_CACHE=/home/xueyang/RLinf/cache/huggingface
export RAY_SCRATCH=/DATA/disk0/xueyang/Data/rlinf-ray
# Set these in the host shell before starting Docker. Never write the API key
# into this repository, a command file, or a committed YAML file.
export WANDB_ENTITY=gxy1000h-jilin-university  # Replace with the W&B user or team name.
export WANDB_PROJECT=pi05-multitask-peft-rl
export WANDB_MODE=online
: "${WANDB_API_KEY:?Export WANDB_API_KEY in the host shell before starting Docker}"
mkdir -p \
  "$GSE_OUTPUT" \
  "$HF_CACHE" \
  "$RAY_SCRATCH/session" \
  "$RAY_SCRATCH/spill" \
  "$RAY_SCRATCH/tmp"
```

Start Docker:

```bash
docker run -it --rm   --privileged \
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
  -e WANDB_API_KEY \
  -e WANDB_ENTITY \
  -e WANDB_PROJECT \
  -e WANDB_MODE \
  -v "$RLINF_REPO":/workspace/RLinf \
  -v "$PI05_SFT":/workspace/models/RLinf-Pi05-MetaWorld-SFT:ro \
  -v "$PI05_RL":/workspace/models/RLinf-Pi05-MetaWorld-RL-FlowSDE:ro \
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

# Append this array to every paper training and evaluation command. Keep
# TensorBoard enabled as a local backup while also streaming all scalar metrics
# and the resolved config to W&B.
WANDB_OVERRIDES=(
  'runner.logger.logger_backends=[tensorboard,wandb]'
  "runner.logger.project_name=${WANDB_PROJECT}"
  "+runner.logger.wandb_entity=${WANDB_ENTITY}"
)

if [[ "${WANDB_MODE}" == "online" ]]; then
  wandb status
fi
```

Use a unique `runner.logger.experiment_name` for every method, training seed,
checkpoint, and evaluation seed. W&B uses this value as the run name. Keep
`runner.logger.log_path` unique as well so TensorBoard, `metrics.jsonl`, W&B
metadata, and console logs cannot overwrite another run. If the server loses
network access, set `WANDB_MODE=offline` before starting Docker; after the run,
sync the generated directory with `wandb sync "$RUN_DIR"/wandb/wandb/offline-run-*`.
W&B is a visualization and comparison backend, not the sole result store:
`metrics.jsonl`, checkpoints, the resolved config, and multi-seed summaries
remain the paper's source of truth.

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
  tests/unit_tests/models/test_openpi_gse.py \
  tests/unit_tests/models/test_openpi_action_lora.py
```

This focused suite passed on 2026-07-17: `45 passed`. The placement and channel
routing tests for parallel rollout separately passed: `47 passed`. Ruff passed
on all changed Python files. Both the training config with task-conditioned
router metrics and the GSE eval-only overrides in Section 6.3 resolved
successfully in the same Docker image.

On 2026-07-30, the three Section 9.13 launch scripts passed `bash -n` and
resolved successfully in `rlinf/rlinf:agentic-rlinf0.3-maniskill_libero` with
their final command lines enforcing micro/global batch `16/1024`, SFT model
paths, `runner.resume_dir=null`, and trainable action adapters. The focused GSE
and OpenPI-GSE tests passed: `27 passed`.

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

Use the validated batch16 values directly on the command line when reproducing
the action-GSE result.

A 320-step run collects 81,920 trajectories, approximately 8.19 million
environment steps, and performs 6,400 optimizer updates. The embodied actor calls
the LR scheduler once per global step, after all PPO minibatches; the historical
`total_training_steps=6400` therefore made the action-GSE LR nearly constant
rather than decaying over 320 steps. Preserve this fact when reproducing the
action-only result. New experiments should set the intended scheduler explicitly.

### 9.2 Formal training command

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
  runner.max_epochs=320 \
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
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

This runs directly to step 320 and evaluates and saves every 20 steps. A
temporarily flat success curve is not sufficient reason to stop because
sparse-reward MT50 evaluations are noisy. Stop only on a hard failure signal from
Section 9.5 or when the one-day compute budget is exhausted.

### 9.3 Resume after interruption

Checkpoint resume restores model, optimizer, and LR scheduler. If the process is
interrupted, point `CKPT` to the latest complete checkpoint and rerun the formal
command with only `runner.resume_dir` changed:

```bash
export CKPT="$RUN_DIR/$EXP_NAME/checkpoints/global_step_100"
test -d "$CKPT/actor"
# Replace runner.resume_dir=null with runner.resume_dir="$CKPT" in Section 9.2
# and append output with: 2>&1 | tee -a "$RUN_DIR/console.log"
```

Do not change `total_training_steps`, GSE architecture, optimizer, batch sizes,
or seed when resuming. With four retained checkpoints, archive any checkpoint
selected as a candidate before later saves prune it.

### 9.4 Live monitoring

TensorBoard:

```bash
tensorboard --logdir "$RUN_DIR/tensorboard" --host 0.0.0.0 --port 6006
```

W&B should show the same scalar namespaces as TensorBoard and
`metrics.jsonl`: `env/*`, `rollout/*`, `train/*`, `eval/*`, and `time/*`.
Immediately after the first global step, verify that the W&B run contains
`env/success_once`, `rollout/rewards`, `train/actor/approx_kl`,
`train/critic/explained_variance`, and `time/step`. For GSE runs, also verify
`train/gse/router/normalized_entropy` and `train/gse/task_router/nmi`. A missing
namespace usually means the wrong Hydra overrides or config profile was used;
do not continue a long run until it is resolved.

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

For the current run, preserve the best checkpoint before checkpoint retention
prunes it:

```bash
export BEST_CKPT=/workspace/output/gse-formal-seed42/gse_formal_seed42/checkpoints/global_step_180
cp -a "$BEST_CKPT" /workspace/output/gse-formal-seed42/best_global_step_180
```

Run fixed-reset evaluation on this checkpoint with three rollout seeds (`42`,
`43`, `44`) before comparing it with the official 70.7% result. Use the eval-only
command in Section 6.3, set `rollout.model.model_path` to
`$BEST_CKPT/actor`, and set `rollout.seed=$SEED`; use a separate `RUN_DIR` for
each seed. Aggregate the resulting `metrics.jsonl` files with the multi-seed
summary tool in Section 9.6.

The following command runs all three evaluations sequentially inside the active
OpenPI Docker shell. It keeps the actor on the original SFT path, because
eval-only rollout workers load the candidate checkpoint directly:

```bash
export BEST_CKPT=/workspace/output/gse-formal-seed42/gse_formal_seed42/checkpoints/global_step_180
test -f "$BEST_CKPT/actor/model_state_dict/full_weights.pt"
export EVAL_ROOT=/workspace/output/gse-step180-multiseed
mkdir -p "$EVAL_ROOT"

for SEED in 42 43 44; do
  export SEED
  export RUN_DIR="$EVAL_ROOT/seed-$SEED"
  mkdir -p "$RUN_DIR"
  python examples/embodiment/train_embodied_agent.py \
    --config-path /workspace/RLinf/examples/embodiment/config \
    --config-name metaworld_50_ppo_openpi_pi05_gse \
    '+model@rollout.model=pi0_5' \
    actor.model.model_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
    rollout.model.model_path="$BEST_CKPT/actor" \
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
    rollout.seed="$SEED" \
    runner.only_eval=True \
    runner.resume_dir=null \
    runner.logger.log_path="$RUN_DIR" \
    runner.logger.experiment_name="gse_step180_seed_$SEED" \
    env.eval.total_num_envs=512 \
    env.eval.use_fixed_reset_state_ids=True \
    env.eval.is_eval=True \
    "${WANDB_OVERRIDES[@]}" \
    2>&1 | tee "$RUN_DIR/console.log"
done

python -m toolkits.embodiment.summarize_multiseed_eval \
  "$EVAL_ROOT"/seed-*/metrics.jsonl \
  --output "$EVAL_ROOT/summary.json"
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

The current candidate is
`$RUN_DIR/$EXP_NAME/checkpoints/global_step_180`; do not overwrite or delete it.
First complete the three-seed evaluation above. Then test one model change at a
time, keeping step-180 GSE as the reference:

1. Add task-wise advantage normalization or capped tail-task weighting to address
   task imbalance; first log per-task counts and advantage scales.

   The logging is implemented: set `algorithm.log_task_advantage_metrics: True`
   and `algorithm.task_advantage_num_tasks: 50` (already enabled by default in
   `examples/embodiment/config/metaworld_50_ppo_openpi_pi05_gse.yaml`) to emit,
   once per global step from `compute_advantages_and_returns`
   (`rlinf/workers/actor/fsdp_actor_worker.py`), per-task PPO sample counts and
   advantage mean/std (`rollout/task_advantage/task_XX/{count,mean,std}` in
   `metrics.jsonl`; per-task keys are suppressed from the terminal table, only
   the summary scalars print). Decision rule from the two summary scalars:
   - Large `rollout/task_advantage/count_cv` (coefficient of variation of
     per-task counts) → per-task sample counts are uneven → prefer **tail-task
     loss weighting**.
   - Large `rollout/task_advantage/std_cv` (coefficient of variation of
     per-task advantage std) while `count_cv` is small → per-task advantage
     *scale* differs, not just sample count → prefer **per-task advantage
     normalization**.
   - Both large → consider combining the two.
   This requires a live rollout batch, so it was not present in the two
   already-completed VLM-GSE evaluations (step 30/step 60); read it from the
   next training run's `metrics.jsonl` before picking between the two options.
   Implementation: `rlinf/utils/metric_utils.py`
   (`compute_task_advantage_metrics`/`_accumulate_task_advantage_stats`/
   `_finalize_task_advantage_metrics`), unit-tested in
   `tests/unit_tests/test_task_advantage_metrics.py`.
2. Compare action-expert GSE against a parameter-matched plain LoRA and a
   no-router multi-expert residual. This isolates the value of orthogonality and
   routing before changing the VLM.
3. Add a small SFT behavior-cloning anchor or reference-policy KL only if the
   multi-seed result shows forgetting. Do not add it preemptively.
4. Tune router temperature/top-k or a small load-balance coefficient only after
   NMI/JS plus per-task deltas show collapse or harmful specialization.
5. Compare GSE against full-parameter PPO and parameter-matched LoRA PPO using
   identical trajectories, seeds, reset states, optimizer updates, and evaluation.
6. After selecting the objective, run at least three training seeds and held-out
   visual/state perturbations before claiming improved generalization.

### 9.8 When to decompose the VLM

The action-only result has plateaued and its three-seed mean remains above SFT,
so an isolated VLM-GSE ablation is now appropriate. Commit `8c59a4a9` supports
upgrading an action-only checkpoint in the correct order: load step-180 action
GSE first, then inject zero-output VLM GSE. The initial upgraded policy is exactly
the step-180 policy.

The first ablation freezes all 126 trained action GSE layers and trains only:

- seven projections in VLM language layer 17 (the final block);
- rank 16 split across four experts (one generalized, three specialized,
  `top_k=2`);
- the existing value head.

The real SFT model contains 1,175,552 trainable VLM-GSE parameters and about
2.39M trainable parameters including the value head. The measured initial VLM
residual difference is exactly zero. Router metrics are separated into
`gse/action_router/*` and `gse/vlm_router/*`; legacy `gse/router/*` continues to
refer to action adapters.


### 9.10 Current method summary

The action-only reference starts from the fully trained Pi0.5 MetaWorld MT50 SFT
checkpoint. The VLM and original action-expert parameters stay frozen. GSE wraps
126 linear projections in the 18-layer action expert with two always-active
generalized experts and six routed specialized experts (`top_k=2`, total rank
64). Orthogonal A matrices and zero B matrices make the initial residual exactly
zero, so the initial policy is identical to SFT.

RLinf uses Flow-SDE/Flow-Noise PPO to collect balanced MT50 trajectories. In the
action-only reference, only action GSE/router parameters and the value head are
optimized. The new branch loads its selected step-180 checkpoint, freezes those
trained action adapters, then adds zero-output GSE to seven projections in the
final VLM language block. Its six-GPU step collects 192 trajectories and runs
four PPO update epochs over five global minibatches; only VLM GSE/router
parameters and the value head update.

Load-balance and orthogonality losses remain disabled; router/task specialization
is monitored rather than forced. The formal objective is parameter-efficient
residual multi-task RL post-training of a frozen SFT VLA, with fixed-reset
per-task evaluation used to detect positive transfer, conflict, and forgetting.

For paper results, report mean, median, worst-5/worst-10, tasks above 90%, negative
transfer relative to SFT, environment steps, trainable parameters, optimizer
memory, peak GPU memory, and wall time. Include held-out visual/state perturbation
tests if making a generalization claim.

### 9.11 Eight-GPU VLM-last-four-layers run

Deliberate expansion beyond the single-block ablation, requested before the
last-block result was matched-evaluated against action-only step 180. Same
upgrade order as 9.8 (load step-180 action GSE, freeze it, add zero-output VLM
GSE) but with `layer_indices=[-4,-3,-2,-1]` (language layers 14-17) instead of
`[-1]`, four times as many wrapped VLM projections (28 instead of 7). Config
lives in a dedicated profile,
`examples/embodiment/config/metaworld_50_ppo_openpi_pi05_gse_vlm_last4.yaml`,
rather than CLI-only overrides, since it is meant to be reused directly.

Rank was deliberately raised beyond a literal copy of the single-block
ablation's hyperparameters. The single-block run used
`total_rank=16`/`num_experts=4`/`num_generalized_experts=1`/`top_k=2`, which
splits to rank 4 per expert (`16 / 4`) — about 0.2% of the VLM's
`hidden_size=2048` and too small to be confident that a null result reflects
the method rather than insufficient capacity. This config instead uses
`total_rank=64`/`lora_alpha=64.0`, matching the action expert's own GSE
capacity (the action expert relies on the unmodified `GSEConfig` defaults of
the same values), keeping `num_experts=4`/`num_generalized_experts=1`/`top_k=2`
unchanged. That splits to rank 16 per expert; with the always-active
generalized expert plus the two routed specialized experts selected by
`top_k=2`, the effective active rank in any one forward pass is `3 * 16 = 48`
out of a rank-64 capacity pool. `lora_alpha` scales with `total_rank` so
`scaling = lora_alpha / total_rank = 1.0` stays the same ratio as the
single-block ablation and the action expert, rather than implicitly
attenuating the larger rank.

Trainable VLM-GSE parameters scale with `total_rank` for the expert factors
plus a `total_rank`-independent router term per projection
(`total_rank * (in_features + out_features) + 3 * in_features`, three routed
experts, no router bias); the single-block ablation's documented 1,175,552
figure at `total_rank=16` and this config's `total_rank=64` both reproduce
exactly from that formula given Gemma-2B's per-projection dimensions. One
`total_rank=64` layer has 4,444,160 trainable VLM-GSE parameters (not a clean
4x of the rank-16 figure, since the router term does not scale with rank);
four layers give 17,776,640, plus the unchanged value head (~1.21M), for about
19M total trainable parameters — still under 1% of the frozen ~3B backbone.

Wrapping four VLM blocks at 4x the per-expert rank of the single-layer
ablation multiplies per-GPU activation memory well beyond the single-layer
ablation, which already OOMed at `micro_batch_size=128` on six GPUs and
required 64. The new config starts conservatively at
`actor.micro_batch_size=16`; treat this as an untested guess, not a validated
value — confirm with a `max_epochs=2` smoke run before trusting a full run,
raise toward 32 only if that smoke run shows headroom, and drop to 8 if 16
still OOMs.

Run inside the OpenPI Docker shell:

```bash
export ACTION_CKPT=/workspace/output/gse-formal-seed42/gse_formal_seed42/checkpoints/global_step_180
test -f "$ACTION_CKPT/actor/model_state_dict/full_weights.pt"
export RUN_DIR=/workspace/output/gse-action180-vlm-last4-seed42
export EXP_NAME=gse_action180_vlm_last4_seed42
mkdir -p "$RUN_DIR"

# Smoke test first: same command, max_epochs=2, save/val checks disabled.
python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse_vlm_last4 \
  actor.model.model_path="$ACTION_CKPT/actor" \
  rollout.model.model_path="$ACTION_CKPT/actor" \
  actor.model.openpi_data.norm_stats_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT/lerobot/metaworld_mt50/norm_stats.json \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="${EXP_NAME}_smoke" \
  runner.max_epochs=2 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee -a "$RUN_DIR/smoke_console.log"

# Verify startup logs report 126 frozen action GSE layers plus 28 trainable
# VLM GSE layers (four blocks x seven projections), action-GSE trainable
# parameters are zero, and the first two PPO updates have finite gradients
# before continuing. If step 3 (micro_batch_size=16) OOMs in the actor
# forward/backward, drop to 8 and repeat the smoke test.

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse_vlm_last4 \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  actor.model.model_path="$ACTION_CKPT/actor" \
  actor.micro_batch_size=16 \
  actor.global_batch_size=1024 \
  env.eval.total_num_envs=512 \
  rollout.model.model_path="$ACTION_CKPT/actor" \
  actor.model.openpi_data.norm_stats_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT/lerobot/metaworld_mt50/norm_stats.json \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=240 \
  runner.save_interval=10 \
  runner.val_check_interval=10 \
  runner.max_checkpoints_to_keep=4 \
  actor.seed=42 \
  rollout.seed=42 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee -a "$RUN_DIR/console.log"
```

To resume this run itself after an interruption (not to reload the action-180
base weights again — `actor.model.model_path`/`rollout.model.model_path`
already handle that on first launch), export `CKPT` to this run's own
checkpoint directory and add `runner.resume_dir="$CKPT"` on the same command.

Compare the selected checkpoint against action-only step 180 with identical
seeds, fixed resets, and the 512-trajectory eight-GPU protocol, using the same
success bar as 9.9/11.2: improve macro success or materially improve worst-10
without a meaningful macro regression. This run does not wait on the
single-layer ablation's result; if it also fails to clear the bar, fall back to
the standing recommendation in 11.2 item 6 (diagnose paired task regressions
and router skew, then try task-wise advantage normalization) rather than
expanding the VLM further.

### 9.12 Seven-GPU action-head plain-LoRA run and evaluation

The completed rank-64 plain-LoRA run uses GPU ranks 1-7. Because the connected
Ray cluster advertises all eight GPUs, explicit component placement is required.
For seven actor ranks and micro batch 128, global batch 896 is the smallest valid
multiple.

```bash
export SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT
export RUN_DIR=/workspace/output/action-lora-r64-seed42
export EXP_NAME=action_lora_r64_seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_action_lora \
  'cluster.component_placement={actor\,env:1-7,rollout:1-7}' \
  env.train.total_num_envs=224 \
  env.train.rollout_epoch=1 \
  rollout.pipeline_stage_num=1 \
  actor.micro_batch_size=128 \
  actor.global_batch_size=896 \
  actor.model.model_path="$SFT_MODEL" \
  rollout.model.model_path="$SFT_MODEL" \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=320 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  runner.max_checkpoints_to_keep=6 \
  env.eval.total_num_envs=448 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  actor.optim.lr=5e-5 \
  actor.optim.total_training_steps=6400 \
  actor.seed=42 \
  rollout.seed=42 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

The selected checkpoint through step 160 is:

```bash
export LORA_CKPT=/workspace/output/action-lora-r64-seed42/action_lora_r64_seed42/checkpoints/global_step_80
test -f "$LORA_CKPT/actor/model_state_dict/full_weights.pt"
```

Evaluate it on the same seven GPUs with 448 fixed-reset trajectories:

```bash
export SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT
export EVAL_DIR=/workspace/output/action-lora-r64-step80-eval-seed42
mkdir -p "$EVAL_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_action_lora \
  'cluster.component_placement={actor\,env:1-7,rollout:1-7}' \
  '+model@rollout.model=pi0_5' \
  actor.model.model_path="$SFT_MODEL" \
  actor.micro_batch_size=128 \
  actor.global_batch_size=896 \
  rollout.model.model_path="$LORA_CKPT/actor" \
  rollout.model.num_action_chunks=5 \
  rollout.model.action_dim=4 \
  rollout.model.is_lora=True \
  rollout.model.lora_rank=64 \
  '+rollout.model.lora_target=action_expert' \
  '+rollout.model.lora_alpha=64.0' \
  '+rollout.model.lora_dropout=0.0' \
  '+rollout.model.lora_init=gaussian' \
  '+rollout.model.lora_target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  '+rollout.model.lora_train_value_head=True' \
  '+rollout.model.lora_require_pi05=True' \
  rollout.model.openpi.config_name=pi05_metaworld \
  rollout.model.openpi.num_images_in_input=1 \
  rollout.model.openpi.noise_method=flow_sde \
  '+rollout.model.openpi_data.norm_stats_path=/workspace/models/RLinf-Pi05-MetaWorld-SFT/lerobot/metaworld_mt50/norm_stats.json' \
  rollout.seed=42 \
  runner.only_eval=True \
  runner.resume_dir=null \
  runner.logger.log_path="$EVAL_DIR" \
  runner.logger.experiment_name=action_lora_r64_step80_eval_seed42 \
  env.eval.total_num_envs=448 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.is_eval=True \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$EVAL_DIR/console.log"
```

### 9.13 Three parallel from-SFT GSE experiments

Three standalone Hydra profiles cover the next comparison. All three load the
original Pi0.5 MT50 SFT checkpoint, set `runner.resume_dir=null`, inject
zero-output GSE, and train the selected adapters from step 0. Do not point any
of these runs at the action-GSE step-180 checkpoint.

| Experiment | Config | Trainable GSE surface | Initial actor micro batch |
|---|---|---|---|
| Action-only rank 32 | `metaworld_50_ppo_openpi_pi05_gse_action_r32` | All 18 action blocks, total rank 32 | 16 |
| Joint action + VLM last 4 | `metaworld_50_ppo_openpi_pi05_gse_joint_vlm_last4` | All action blocks plus VLM language blocks 14-17, rank 64 | 16 |
| Joint action + all VLM layers | `metaworld_50_ppo_openpi_pi05_gse_joint_vlm_all` | All action blocks plus all 18 VLM language blocks, rank 64 | 16 |

Here, "all VLM layers" means GSE on the seven target linear projections in all
18 language Transformer blocks. The original VLM/action weights and visual
encoder remain frozen; it does not mean full-parameter VLM PPO. Both action and
VLM adapters/routers train jointly in the two joint profiles. The rank-32 action
profile sets `total_rank=32` and `lora_alpha=32`, so eight experts receive rank
4 each while preserving unit adapter scaling.

The commands below assume the Docker initialization and `WANDB_OVERRIDES` from
Section 7. The launch scripts explicitly enforce
`actor.micro_batch_size=16` and `actor.global_batch_size=1024` and forward any
additional Hydra overrides. Use one isolated eight-GPU job/container per command
when launching them concurrently.

Action-only rank 32:

```bash
bash examples/embodiment/run_metaworld_gse_action_r32.sh \
  "${WANDB_OVERRIDES[@]}"
```

Joint action + VLM-last4 from SFT:

```bash
RUN_DIR=/workspace/output/gse-joint-vlm-last4-smoke \
EXP_NAME=gse_joint_vlm_last4_smoke \
bash examples/embodiment/run_metaworld_gse_joint_vlm_last4.sh \
  runner.max_epochs=2 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  "${WANDB_OVERRIDES[@]}"

# Launch the full run only after the smoke run is healthy.
bash examples/embodiment/run_metaworld_gse_joint_vlm_last4.sh \
  "${WANDB_OVERRIDES[@]}"
```

Joint action + all 18 VLM language layers from SFT:

```bash
RUN_DIR=/workspace/output/gse-joint-vlm-all-smoke \
EXP_NAME=gse_joint_vlm_all_smoke \
bash examples/embodiment/run_metaworld_gse_joint_vlm_all.sh \
  runner.max_epochs=2 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  "${WANDB_OVERRIDES[@]}"

# Launch the full run only after the smoke run is healthy.
bash examples/embodiment/run_metaworld_gse_joint_vlm_all.sh \
  "${WANDB_OVERRIDES[@]}"
```

Both joint profiles intentionally start at micro batch 16. If an all-layer
smoke run OOMs, change that profile/script deliberately and record the resulting
gradient-accumulation difference; do not silently alter the 1,024 global batch.

For both joint runs, startup must report 126 action GSE layers plus 28 VLM GSE
layers for last4 or 126 VLM GSE layers for all18. Confirm that action and VLM
adapter trainable counts are both nonzero, the first two PPO updates have finite
loss/gradient values, and the initial residual remains zero before committing a
long run. Keep the same 128 environments x 2 rollout epochs, 1,024 global batch,
fixed 512-trajectory evaluation, PPO settings, seeds, and checkpoint selection
protocol when comparing the three methods.

## 10. Development cautions

- Read the repository-root `AGENTS.md` before editing.
- Preserve unrelated user changes and real server paths.
- Keep the official full-parameter Pi0.5 PPO config unchanged as a baseline.
- Keep the staged action-180 VLM ablations distinct from the new from-SFT joint
  experiments; never resume one protocol from a checkpoint of the other.
- Zero-B initialization delays useful A/router gradients until B becomes nonzero.
- Aggregate load balance over a diverse multi-task batch; per-sample balancing can
  be meaningless or harmful.
- Do not claim specialization from high entropy or uniform aggregate usage alone.
- Do not terminate unrelated GPU jobs or delete shared Docker/Ray data.
- Put checkpoints, TensorBoard data, videos, and large logs outside the repository.

## 11. Next-conversation handoff

### 11.1 Current run state

Three new from-SFT experiments are ready to launch in parallel from Section
9.13: action-only rank 32, joint action+VLM-last4, and joint action+all18 VLM
language layers. All three deliberately use actor micro/global batch `16/1024`;
run the two-step smoke commands for both joint profiles before long training.
These experiments are separate from the staged action-180 VLM runs below.

A second, deliberately expanded experiment was added on top of this one before
its matched-evaluation result came back: `gse-action180-vlm-last4-seed42`
(9.11), which widens `layer_indices` from `[-1]` to `[-4,-3,-2,-1]`, raises
`total_rank` from 16 to 64 (matching the action expert's own GSE capacity,
judged necessary since rank 4 per expert was too small to trust a null result
as a capacity-independent finding), and moves from a six-GPU constrained
placement to the full eight-GPU cluster. Its `actor.micro_batch_size=16` is an
untested guess based on the combined four-block-count and 4x-rank memory
increase over the single-block ablation, not a validated value — run the
`max_epochs=2` smoke test in 9.11 before trusting a full run. Evaluate it
against action-only step 180 using the established 512-trajectory eight-GPU
protocol, same as any other eight-GPU candidate.

The first (six-GPU, single-block) active experiment is
`gse-action180-vlm-last-seed42`. It starts from the
selected action-GSE step-180 checkpoint, freezes all action adapters, and trains
the seven GSE-wrapped projections in VLM language layer 17 plus the value head.
The six-GPU run uses explicit placement on ranks 0-5, 192 trajectories per
global step, `rollout_epoch=1`, actor global batch 768, and actor micro batch 64.
Micro batch 128 OOMed in the actor forward at the VLM MLP `up_proj` GSE
specialized expert; micro batch 64 runs successfully and gives two-way gradient
accumulation per rank.

The first successful training-step metrics were healthy:

```text
approx_kl=2.60e-4, clip_fraction=0.0015, grad_norm=0.497
critic explained_variance=0.787, value_loss=0.016
action router normalized_entropy=0.954
VLM router entropy=0.754, normalized_entropy about 0.686
VLM specialized selection about [0.499, 0.217, 0.284]
```

The VLM router is skewed but not collapsed; all three specialized experts still
receive traffic. Do not enable a balancing loss from one step. Track whether the
same expert remains selected almost always over multiple checkpoints. The
training-rollout `success_once=65.63%` came from only 192 trajectories and is not
comparable to the action-only three-seed result.

The command currently uses `env.eval.total_num_envs=96`. Treat this only as a
runtime health check: it gives fewer than two trials per MT50 task and makes
worst-5/worst-10 and tasks-above-90 statistically unusable. For checkpoint
selection, run a separate matched evaluation with 600 trajectories on six GPUs
(12 per task), or use the established 512-trajectory protocol for both candidate
and baseline on eight GPUs. When starting fresh, set `runner.resume_dir=null`;
when resuming, explicitly export `CKPT` to the runner checkpoint directory before
using `runner.resume_dir="$CKPT"`.

### 11.2 Work to do when results arrive

1. Identify the evaluated checkpoint and confirm whether the run was fresh or
   resumed. Record wall time, peak GPU memory, trajectories, optimizer updates,
   and exact seeds.
2. Parse `metrics.jsonl` across all available steps. Check success-once, macro,
   worst-10, tasks above 90%, KL, clip fraction, gradient norm, critic explained
   variance, and both action/VLM router statistics. Do not select a checkpoint
   from one noisy evaluation.
3. Select the best provisional VLM checkpoint using repeated fixed-reset
   evaluations, then evaluate it and action-only step 180 with identical seeds,
   reset states, trajectory counts, Flow-SDE settings, and success definitions.
   Report paired per-task deltas and the multi-seed confidence interval.
4. Consider VLM GSE successful only if the matched comparison improves macro
   success or materially improves worst-10 without a meaningful macro regression.
   The action-only reference is `72.72%` success-once and `72.73%` macro over
   three seeds, with `18.73%` worst-10.
5. If final-block VLM GSE succeeds, next compare action-only, frozen-action plus
   VLM GSE, and joint action+VLM GSE with separate learning rates. Add language
   layer 16 only after the final-block result is reproduced over multiple seeds.
6. If it does not succeed, do not expand the VLM. First diagnose paired task
   regressions and router skew. The next primary method experiment is task-wise
   advantage normalization; an attention-only VLM target
   `[q_proj,k_proj,v_proj,o_proj]` is a separate memory/architecture ablation.
7. The first parameter-matched plain-LoRA run is complete (Section 5.5/9.12).
   Re-evaluate it under the final matched protocol, then run the no-router
   multi-expert and full-parameter controls with the same data budget, followed
   by at least three training seeds and held-out visual/state perturbations.

For eval-only loading of a joint action+VLM checkpoint, point
`rollout.model.model_path` directly at its saved actor directory and mirror the
complete GSE configuration under `rollout.model.gse`; eval-only mode does not
synchronize actor weights to rollout workers.

### 11.3 Git requirement for future work

After every completed code/config/test/documentation stage, run the focused
validation and automatically create a signed Conventional Commit with
`git commit -s`; do not wait for another user reminder. Inspect `git status`
before staging, preserve unrelated user changes, never include checkpoints or
large outputs, and do not push, rebase, reset, or clean the worktree unless the
user explicitly requests it. Record each new commit and its validation result in
this handoff.

The immediate milestone for the next conversation is to smoke-test and launch
the three Section 9.13 from-SFT experiments, while continuing to ingest the
staged `gse-action180-vlm-last4-seed42` result if it arrives. Keep the two
protocols separate, select checkpoints without treating health-check evaluations
as paper evidence, and keep auxiliary losses disabled until matched results show
that router skew is persistent and harmful.
