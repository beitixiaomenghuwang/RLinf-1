# Pi0.5 Multi-Task Parameter-Efficient RL Handoff

Last updated: 2026-08-21 (Section 12.7: current LIBERO-90 PE-RL freezes the
vision backbone and trains semantic-conditioned projector/LLM GSE; all
OpenVLA-OFT runs use SDPA)

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
- orthogonal A subspaces and zero-initialized B matrices (the default GSE
  initialization);
- generalized experts are always active;
- specialized experts use a learned sparse router.

The rank-32 SVD ablation uses a separate policy-preserving parameterization:
each frozen linear weight is decomposed with a complete float32 SVD, all
leading rank-32 A/B factors are trainable, and a non-persistent frozen copy of
those initial factors is subtracted in the residual forward. This follows the
MoORE dynamic-minus-baseline construction, so the initial policy remains
exactly the loaded SFT policy while both factor matrices train from step 0.

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

On 2026-07-30, the two Section 9.13 config profiles and three terminal overrides
resolved successfully in `rlinf/rlinf:agentic-rlinf0.3-maniskill_libero` with
micro/global batch `16/1024`, SFT model paths, `runner.resume_dir=null`,
trainable action adapters, and a 320-step cosine schedule. The focused GSE and
OpenPI-GSE tests passed: `27 passed`.

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
the same values), with `num_experts=8`/`num_generalized_experts=1`/`top_k=2`
and token-level routing. That splits to rank 8 per expert; with the always-active
generalized expert plus the two routed specialized experts selected by
`top_k=2`, the effective active rank in any one forward pass is `3 * 8 = 24`
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

### 9.13 Four parallel from-SFT GSE experiments

Four experiments using three reusable Hydra profiles cover the next comparison.
All four load the original Pi0.5 MT50 SFT checkpoint, set
`runner.resume_dir=null`, inject zero-output GSE, and train the selected adapters
from step 0. Do not point any of these runs at the action-GSE step-180
checkpoint.

| Experiment | Config | Trainable GSE surface | Initial actor micro batch |
|---|---|---|---|
| Action-only rank 32 | `metaworld_50_ppo_openpi_pi05_gse_action_r32` | All 18 action blocks, total rank 32 | 16 |
| Action-only rank 32 + per-task advantage | `metaworld_50_ppo_openpi_pi05_gse_action_r32_per_task_adv` | Same action-only rank-32 surface with per-task GAE normalization | 16 |
| Action-only rank 32 + per-task + 1G/7S | `metaworld_50_ppo_openpi_pi05_gse_action_r32_per_task_adv_1g7s` | Same surface, one generalized and seven specialized experts | 16 |
| Joint action + VLM last 4 | `metaworld_50_ppo_openpi_pi05_gse_joint_vlm` | All action blocks plus VLM language blocks 14-17, rank 64 | 16 |
| Joint action + all VLM layers | `metaworld_50_ppo_openpi_pi05_gse_joint_vlm` | All action blocks plus all 18 VLM language blocks, rank 64 | 16 |

Here, "all VLM layers" means GSE on the seven target linear projections in all
18 language Transformer blocks. The original VLM/action weights and visual
encoder remain frozen; it does not mean full-parameter VLM PPO. Both action and
VLM adapters/routers train jointly in the two joint runs. The rank-32 action
profile sets `total_rank=32` and `lora_alpha=32`, so eight experts receive rank
4 each while preserving unit adapter scaling.

The main comparison uses no LR warmup. Zero-output initialization already starts
from the exact SFT policy, and the validated action-GSE run did not require a
warmup. Add a 5% warmup only as a separate stability ablation if the first
checkpoints show excessive KL, clip fraction, or gradient norm. The historical
action-GSE command used cosine decay with `total_training_steps=6400` for only
320 scheduler calls, so the LR retained about 99.4% of its initial value; this
was effectively a constant LR and likely contributed to the observed lack of
late-stage settling. New experiments use a half-cycle cosine schedule with
`num_cycles=0.5` and keep `total_training_steps` exactly synchronized with
`runner.max_epochs`, so the LR decays across the full run and approaches zero at
the final checkpoint.

This scheduler correction applies only to fresh launches with
`runner.resume_dir=null`. Resuming a checkpoint restores its saved optimizer and
scheduler state; a run started with constant LR or the historical 6,400-step
horizon must be restarted from SFT rather than resumed if it is intended to use
the corrected full-run cosine decay.

The commands below assume the Docker initialization and `WANDB_OVERRIDES` from
Section 7. All important experiment parameters are expanded as terminal
overrides so they can be edited in one place. Use one isolated eight-GPU
job/container per command when launching them concurrently.

Action-only rank 32:

```bash
export SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT
export RUN_DIR=/workspace/output/gse-action-r32-seed42
export EXP_NAME=gse_action_r32_seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse_action_r32 \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=1024 \
  actor.model.model_path="$SFT_MODEL" \
  rollout.model.model_path="$SFT_MODEL" \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=240 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  runner.max_checkpoints_to_keep=4 \
  env.eval.total_num_envs=512 \
  actor.seed=42 \
  rollout.seed=42 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

The matched Per-task advantage ablation uses the same command and hardware
settings, with only these run-specific substitutions:

```bash
--config-name metaworld_50_ppo_openpi_pi05_gse_action_r32_per_task_adv
runner.logger.experiment_name=gse_action_r32_per_task_seed42
```

Both profiles keep the default action learning rate (`5e-6`) and value learning
rate (`1e-4`) and use a 240-step cosine horizon. The Per-task profile normalizes
GAE advantages independently for the 50 MetaWorld tasks, with global-statistics
fallback for sparse or zero-variance tasks. It also logs aggregate and
per-router-layer task information metrics; use
``gse/task_router/adjusted_cramers_v``,
``gse/task_router/prob_nmi``, and the ``layerwise_*_max`` metrics to distinguish
true absence of detectable task information from cancellation across layers.

The fixed-rank 1G/7S action ablation uses the same command with
`--config-name metaworld_50_ppo_openpi_pi05_gse_action_r32_per_task_adv_1g7s`.
It remains a 240-step run with the same learning rates and per-task
normalization; only the generalized/specialized expert split changes.

Joint action + VLM-last4 from SFT:

```bash
export SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT
export RUN_DIR=/workspace/output/gse-joint-vlm-last4-seed42
export EXP_NAME=gse_joint_vlm_last4_seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse_joint_vlm \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=1024 \
  actor.model.model_path="$SFT_MODEL" \
  rollout.model.model_path="$SFT_MODEL" \
  actor.model.gse.train_action_adapters=True \
  'actor.model.gse.vlm.layer_indices=[-4,-3,-2,-1]' \
  actor.model.gse.vlm.total_rank=64 \
  actor.model.gse.vlm.lora_alpha=64.0 \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=320 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  runner.max_checkpoints_to_keep=4 \
  env.eval.total_num_envs=512 \
  'actor.optim.total_training_steps=${runner.max_epochs}' \
  actor.optim.lr_warmup_steps=0 \
  actor.optim.lr_scheduler=cosine \
  actor.optim.num_cycles=0.5 \
  actor.seed=42 \
  rollout.seed=42 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Joint action + all 18 VLM language layers from SFT:

```bash
export SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT
export RUN_DIR=/workspace/output/gse-joint-vlm-all-seed42
export EXP_NAME=gse_joint_vlm_all_seed42
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name metaworld_50_ppo_openpi_pi05_gse_joint_vlm \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=1024 \
  actor.model.model_path="$SFT_MODEL" \
  rollout.model.model_path="$SFT_MODEL" \
  actor.model.gse.train_action_adapters=True \
  'actor.model.gse.vlm.layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]' \
  actor.model.gse.vlm.total_rank=64 \
  actor.model.gse.vlm.lora_alpha=64.0 \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  runner.max_epochs=320 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  runner.max_checkpoints_to_keep=4 \
  env.eval.total_num_envs=512 \
  'actor.optim.total_training_steps=${runner.max_epochs}' \
  actor.optim.lr_warmup_steps=0 \
  actor.optim.lr_scheduler=cosine \
  actor.optim.num_cycles=0.5 \
  actor.seed=42 \
  rollout.seed=42 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Both joint runs intentionally start at micro batch 16. If an all-layer
smoke run OOMs, change the terminal override deliberately and record the
resulting gradient-accumulation difference; do not silently alter the 1,024
global batch. For a two-step smoke run, change `runner.max_epochs=320` to `2`
and set both `runner.save_interval=-1` and `runner.val_check_interval=-1`.

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

## 12. LIBERO-90 OpenVLA-OFT GSE 八卡训练(2026-08-13)

本节是 [PARALLEL_SERVER_HANDOFF.md](PARALLEL_SERVER_HANDOFF.md) 第 14 节
"LIBERO-90 OpenVLA-OFT GSE 四卡训练"在本机(8×A100-SXM4-80GB,主机内存
1 TB)上的对应最大训练指令。四卡文档记录的是 4×A100-PCIE-40GB 上实测的
最大稳定配置。本节的并行度配置**已在本机实测定稿**(2026-08-13):最初按
"GPU 数翻倍、单卡显存翻倍"外推的若干值在实测中被推翻,表中记录的是最终
实测值与被否决的候选值,不要再按纸面外推调整。2026-08-13 的吞吐/显存
测量来自旧 LLM-only GSE；12.5d 记录的 whole-model GSE 也已被 12.7 的
frozen-vision 方法定义取代。旧吞吐和显存只作历史诊断，当前基线必须重新测量。

实测定稿:`128 env × 2 epoch` 训练、`64 env × 8 epoch` 评估、
`rollout.micro_batch_size=8`、`actor.micro_batch_size=32`、allocator 仅设
`max_split_size_mb:128`。环境并行度已验证；frozen-vision 模型的 actor
吞吐和显存尚未完成正式八卡测量。whole-model 的 1107.184 s/step 和 57 GiB
峰值不得用于新实验工期估算。

### 12.1 入口与资产

| 项目 | 值 |
|---|---|
| Hydra 配置 | `examples/embodiment/config/libero_90_grpo_openvlaoft_gse_r32_svd.yaml` |
| Docker 镜像 | `rlinf/rlinf:agentic-rlinf0.3-maniskill_libero`(本机已存在) |
| 容器内环境 | `source switch_env openvla-oft` |
| base SFT 权重 | 宿主 `/DATA/disk0/xueyang/model/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora/`(本地 NVMe),以只读方式挂载到容器内 `${REPO_PATH}/model/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora/`,使 YAML 中的 `${oc.env:REPO_PATH}` 模型路径无需覆盖即可解析 |
| 本机 GPU / 内存 | 8×A100-SXM4-80GB(81,920 MiB/卡)/ 1007 GiB RAM |

启动前必须先确认宿主模型目录中的 `model.safetensors.index.json` 与四个
`model-0000X-of-00004.safetensors` 分片存在。这些分片是完整权重,训练必须
保持 `actor.model.is_lora=false`,不得再次合并或加载 `lora_adapter`(与四卡
协议一致)。GSE 结构保持 YAML 默认:8 专家(1 generalized + 7 specialized)、
`top_k=2`、total rank 32、SVD 初始化。

**模型权重必须放在本地磁盘,严禁直接挂载 `/ks3` 对象存储目录。**
2026-08-13 首次启动曾把 `/ks3/guoxueyang/model/...`(`fuse.ks3fs` 挂载)直接
bind 进容器:`from_pretrained` 以 mmap 方式读取 safetensors 分片,ks3fs 的
缺页读取失败使全部 8 个 rollout worker 在模型加载中同时收到 SIGBUS
(`Fatal Python error: Bus error`),并伴随 raylet 因内存压力 OOM-kill 2 个
worker,训练在 step 0 之前退出。权重已用 `rsync` 复制到上表的本地 NVMe
路径;后续任何从 `/ks3` 新下载的模型都必须先复制到本地再挂载。

### 12.2 八卡最大训练配置(与四卡实测对照)

| 项目 | 四卡 40G 实测值 | 八卡 80G 实测定稿 | 状态 |
|---|---:|---:|---|
| 训练并行度 | `64 env × 4 rollout epoch` | `128 env × 2 rollout epoch` | 协议等价(均为 256 条轨迹/step、32 个 group)。**`256 env × 1` 已被实测否决**:32 env/GPU 时 env worker 初始化即 `EGL_NOT_INITIALIZED`;16 env/GPU 是本机 EGL 上限 |
| LIBERO-90 周期评估 | `32 env × 16 epoch` | `64 env × 8 epoch` | 总量同为 512 个 fixed 窗口。同理不能用 `512 env × 1`(EGL 上限);必须显式设 `env.eval.rollout_epoch=8`,否则 YAML 默认 16 会使每次评估跑 8192 条轨迹 |
| actor micro / global batch | `16 / 1024` | `32 / 16384` | global batch 与当前官方协议一致；frozen-vision 首步需要重新验证。micro 64 的 +92 s 退化来自旧 LLM-only 调优 |
| rollout micro batch | `4` | `8` | 8 是旧 LLM-only 实测最优值；frozen-vision 首步需重新记录实际吞吐 |
| actor backend | FSDP + gradient checkpointing | 同左 | 沿用 |
| actor / rollout offload | 均开启 | 均开启 | 沿用；57 GiB 是旧 whole-model 数据，frozen-vision 峰值待重测 |
| PyTorch allocator | `max_split_size_mb:128,garbage_collection_threshold:0.8` | **仅** `max_split_size_mb:128` | **必须删掉 `garbage_collection_threshold:0.8`**:本机带该项时 rollout worker 在生成中 `SIGSEGV`(GC 回收与 offload 的段释放竞争),移除后连续多步无异常。`expandable_segments` 未实测,不要默认打开 |
| 训练步数 / 保存 / 评估 | `120 / 10 / 10` | `140 / 10 / 10` | 步数延长为 140,必须通过 `runner.max_steps=140` 设置(runner 取 `max_epochs` 换算值与 `max_steps` 的较小者,YAML 中 `max_steps: 120` 不覆盖则 120 步即停);cosine horizon 经 `total_training_steps=${runner.max_steps}` 自动跟随为 140 |

八卡与四卡每个 global step 的轨迹数(256)和 GRPO group 结构(32 组 × 8)
一致；当前 actor global batch 使用官方值 16384。训练步数从 120 延长为
140，scheduler horizon 随之变为 140；旧四卡曲线只能作为历史参考，不能
与改变模型结构后的 frozen-vision 曲线直接归因比较。

### 12.3 Docker 启动

在宿主 shell 中(沿用第 7 节的 W&B 约定,不要把 API key 写入仓库):

```bash
export RLINF_REPO=/home/xueyang/RLinf
# 必须是本地磁盘路径;不要指向 /ks3(FUSE 对象存储,mmap 读取会 SIGBUS)。
export OFT_MODEL=/DATA/disk0/xueyang/model/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
export OFT_OUTPUT=/DATA/disk0/xueyang/model/openvlaoft-libero90-gse
export HF_CACHE=/home/xueyang/RLinf/cache/huggingface
export RAY_SCRATCH=/DATA/disk0/xueyang/Data/rlinf-ray
export WANDB_ENTITY=gxy1000h-jilin-university
export WANDB_PROJECT=pi05-multitask-peft-rl
export WANDB_MODE=online
test -f "$OFT_MODEL/model.safetensors.index.json"
mkdir -p "$OFT_OUTPUT" "$HF_CACHE" "$RAY_SCRATCH/session" "$RAY_SCRATCH/spill" "$RAY_SCRATCH/tmp"

docker run -it --rm --privileged \
  --gpus all \
  --shm-size 256g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name rlinf-openvlaoft-libero90 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e RLINF_RAY_TEMP_DIR=/workspace/ray/session \
  -e RLINF_RAY_OBJECT_SPILL_DIR=/workspace/ray/spill \
  -e TMPDIR=/workspace/ray/tmp \
  -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e WANDB_MODE \
  -v "$RLINF_REPO":/workspace/RLinf \
  -v "$OFT_MODEL":/workspace/RLinf/model/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora:ro \
  -v "$OFT_OUTPUT":/workspace/output \
  -v "$RAY_SCRATCH":/workspace/ray \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /workspace/RLinf \
  rlinf/rlinf:agentic-rlinf0.3-maniskill_libero \
  bash
```

容器内初始化(注意环境是 `openvla-oft`,不是 `openpi`):

```bash
source switch_env openvla-oft
cd /workspace/RLinf

export REPO_PATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# 不要加 garbage_collection_threshold:0.8 —— 本机实测会让 rollout worker SIGSEGV。
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

test -f "$REPO_PATH/model/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora/model.safetensors"

WANDB_OVERRIDES=(
  'runner.logger.logger_backends=[tensorboard,wandb]'
  "runner.logger.project_name=${WANDB_PROJECT}"
  "+runner.logger.wandb_entity=${WANDB_ENTITY}"
)
```

只用 TensorBoard 时删去 `"${WANDB_OVERRIDES[@]}"` 即可。


### 12.5 正式最大训练指令(140 步)

> **2026-08-21 更新：** 当前 PE-RL 协议冻结视觉 backbone，以下命令和默认
> 实验名已切换到 frozen-vision 配置。12.5d 中的 whole-model 437 层运行是
> 历史结果，只能用于诊断，禁止从其 checkpoint resume。

```bash
export RUN_DIR=/workspace/output/libero90_gse_frozen_vision_r32_svd_8gpu_seed1234
mkdir -p "$RUN_DIR"

python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name libero_90_grpo_openvlaoft_gse_r32_svd \
  env.train.total_num_envs=128 \
  env.train.rollout_epoch=2 \
  env.eval.total_num_envs=64 \
  env.eval.rollout_epoch=8 \
  rollout.micro_batch_size=8 \
  actor.micro_batch_size=32 \
  actor.global_batch_size=16384 \
  runner.max_steps=140 \
  +runner.max_checkpoints_to_keep=6 \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name=libero90_gse_frozen_vision_r32_svd_8gpu_seed1234 \
  "${WANDB_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/console.log"
```

两个必须显式设置的覆盖不要省略:训练步数由 `runner.max_steps` 控制
(runner 取 `max_epochs` 换算值与 `max_steps` 的较小者,只改
`runner.max_epochs` 时 YAML 中的 `max_steps: 120` 仍会使训练在 120 步停止);
评估必须带 `env.eval.rollout_epoch=8`,否则 YAML 默认的 16 轮会让每次评估
执行 `64×16=1024` 条轨迹。

`rollout.micro_batch_size=8` 也不能省略:YAML 默认 4 是为 40 GiB 卡设定的,
在本机会让每 step 多花约 60 s。

**推荐用自动续跑循环脚本代替裸命令**:
[scripts/run_libero90_gse_8gpu_loop.sh](scripts/run_libero90_gse_8gpu_loop.sh)
内置上面全部覆盖,训练异常退出时自动从最新完整 checkpoint(带 `actor/`
目录)续跑,并设 `+runner.eval_on_resume=true` 补做落在验证间隔上的评估:

```bash
# 容器内,完成 12.3 的初始化后:
export RUN_DIR=/workspace/output/libero90_gse_frozen_vision_r32_svd_8gpu_seed1234
mkdir -p "$RUN_DIR"
bash scripts/run_libero90_gse_8gpu_loop.sh
```

### 12.5a LIBERO env 主机内存泄漏与子进程重生补丁(2026-08-13)

首次长跑在 step 8 的 rollout 中被 Ray memory monitor 杀死:主机内存从
init 后的约 500 GB 单调涨到 964 GB(95.7% > 默认阈值 95%)。逐进程排查
显示泄漏不在 8 个 rollout/actor 大进程(RSS 稳定),而在约 200 个 LIBERO
env 子进程:每条轨迹结束后按新任务 reconfigure 时,旧代码在**同一个长命
子进程内** `env.close()` 后直接重建 `OffScreenRenderEnv`,robosuite/MuJoCo
在 close 后遗留原生内存,每次任务切换每进程涨约 100–140 MB,合计约
30–80 GB/step。在 close 后加 `gc.collect()` 实测无效(原生泄漏,不在
Python 堆上)。

修复:`rlinf/envs/libero/venv.py` 的
`ReconfigureSubprocEnvWorker.reconfigure_env_fn` 改为终止旧子进程并 spawn
新进程,RSS 随进程退出归零。实测每 step 增加约 25–45 s(子进程重生开销),
主机内存曲线由单调上升变为平稳:step 0/2 完成后均约 487–489 GB,斜率为零。
同时将 `RAY_memory_usage_threshold=0.98` 作为兜底(1 TB 机器上 95% 阈值
留白过大),配合 12.5 的自动续跑循环,即使再次触发保护也能从 checkpoint
恢复而不是死掉。

注意:`runner.save_interval` 必须能被 `val_check_interval` 整除的断言方向
是 `save % val == 0`;想加密 checkpoint(如每 5 步)必须同时把两者设为 5,
只改 save_interval 会在启动即断言失败。

其余全部沿用 YAML 内已固化的正式协议:每 10 步保存并评估、
`save_best_macro_mean=True`、GRPO `group_size=8`、温度 1.6 随机评估、
GSE 1G+7S rank 32 SVD、全量 task-router 指标(`task_router_num_tasks: 90`)。
中断续跑时保持全部参数不变,仅设
`runner.resume_dir="$RUN_DIR/libero90_gse_frozen_vision_r32_svd_8gpu_seed1234/checkpoints/global_step_<N>"`。

### 12.5b 评估低成功率根因:flash_attention_2 在本机行为级损坏(2026-08-13)

> **历史诊断说明:** 本节关于 FA2 损坏的结论仍成立，但其中把 SDPA 下
> 55.9%/71.5--75.0% 的差距归因于平台差异的结论已被 12.5d 的严格对照
> 推翻。这些低数值使用了原始环境图像和/或 greedy，并非官方发布协议，
> 不得进入论文结果表。

训练期 greedy 周期评估在 step 10/20 恒为 3.4–3.6%,与训练 rollout 的高温
采样成功率(3–7%)同量级,而并行文档 14.6 节记录同语义 greedy 评估在 ARM
上为 96.3%。经过完整对照排查,根因是 **`attn_implementation:
"flash_attention_2"`(模型组默认值)对 OpenVLA-OFT 推理是普遍性行为损坏
——不分平台**。2026-08-14 已在 ARM 上用同一 sha256 校验过的 OFT fork
transformers(4.40.1)复测:启用 FA2 后 greedy 同样崩到 2.34%。ARM 此前
未暴露只因其运行时未走 FA2 路径。**结论:所有 OpenVLA-OFT 训练与评估、
在任何机器上,都必须显式 `attn_implementation=sdpa`(或 eager),严禁
使用模型组默认的 flash_attention_2。**两机训练/评估配置已同步改为 sdpa。
诊断矩阵(本机、greedy、同一批 fixed 有序 reset 窗口):

| 模型(权重均已 sha256 对照 HuggingFace 原版一致) | flash_attention_2 | sdpa | eager |
|---|---:|---:|---:|
| Base-Lora SFT(官方口径 42.67%) | 4.5%(512 条) | **55.9%**(256 条) | — |
| GRPO RL(README 96.44%;ARM greedy 96.3%) | 4.3%(256 条) | **71.5%**(256 条) | 73.1%(256 条) |

排查中同时排除的假设(全部有实证):权重损坏(两模型 8 个分片 sha256
均与 HF 一致)、采样模式未切换(resolved config `temperature_eval: -1` 且
worker/模型代码 greedy 分支为 argmax)、图像方向(`get_libero_image` 有
180° 旋转)、物体未稳定(reset 后 15 步零动作)、夹爪符号(变换与
openvla 官方评估一致)、GSE/权重同步(base 模型独立评估同样低)、
`lora_adapter` 未合并(已由并行文档 14.6 排除,不得合并)。视频抽帧显示
flash-attn 下策略仍是目标导向(能精准接近任务物体)但完成不了操作——
高温采样时两机成功率一致(都被温度抹平)、greedy 相差 20 倍,正是
attention 实现损坏的典型特征。

因此本机所有 OpenVLA-OFT 训练与评估必须覆盖
`actor.model.attn_implementation=sdpa`(sdpa 与 eager 在 256 条内等价,
sdpa 更快;循环脚本已内置该 override)。**正式训练已于 2026-08-14 从 0
以全程 sdpa 重启**:此前 step 0–20 在损坏的 FA2 下训练、step 21–40 为
sdpa 的混合协议运行已整体归档为
`libero90_gse_r32_svd_8gpu_seed1234_mixed_fa2_20steps_sdpa_40steps`
(其 sdpa 补评曲线 base 55.9% → step-20 59.0% → step-30 61.5% →
step-40 59.4% 说明 FA2 时代的 GSE 残差并未损害 sdpa 推理,可作探索性
参考,但不进论文)。诊断产物保留在 `OFT_OUTPUT` 下的 `diag_*` 目录;
官方 GRPO 模型位于 `$OFT_OUTPUT/RLinf-OpenVLAOFT-GRPO-LIBERO-90/`
(sha256 已校验)可复用。

**修复后与 ARM 仍存的 ~20–25 pp 系统性差距**(GRPO:ARM 96.3% vs 本机
75.0%;base:ARM ~80% vs 本机 55.9%)已按六项假设逐一核对
(2026-08-14):

1. 指标口径:排除。本机报告的就是 `eval/success_once`(prompt50 run:
   once 74.6% / at_end 68.8%,两字段均已取出),不是 at_end 误读。
2. 采样模式:排除。resolved config `temperature_eval: -1`,worker 映射为
   `do_sample=False` 纯 greedy。
3. 多轮 fixed-reset 复用:排除。所有多轮评估 `covered_tasks=90`;若存在
   首批 ID 重复 bug,64-env 轮次只能覆盖 64 任务。本仓库含 LIBERO 池推进
   实现(并行文档 14.4 协议)。
4. 归一化键:排除。resolved `unnorm_key=libero_90_no_noops_trajall`。
5. 加载语义:排除。`is_lora=false`、`num_action_chunks=8`、bf16;
   `max_prompt_length` 128→50 仅 71.5%→75.0%。
6. 环境侧:**定位在此**。配置逐项一致(512 步、256×256、standard、
   `reset_gripper_open=False`、EGL 正常),但 16 条 fixed-reset 严格对照
   (task 0–15 trial 0,greedy,ARM 同协议 100%)本机只有
   **68.75%(11/16)**,失败任务 {1, 6, 8, 11, 12} 与 256 条评估中的
   部分失败任务完全重合——失败指纹稳定,是仿真物理/渲染栈版本差异,
   不是统计噪声。

本机容器栈为 `mujoco 3.9.0 / robosuite 1.4.1 / libero 0.1.0 /
transformers 4.40.1 / torch 2.6.0`。**2026-08-14 已与 ARM 实际生效栈
逐项对齐核对**:ARM 生效的是 `mujoco 3.9.0`(venv 3.11.0 被 PYTHONPATH
前置的 libero-runtime 遮蔽)、`robosuite 1.4.1`、LIBERO 源码 commit
`0c5e40c`、`PyOpenGL 3.1.10`——与本机全部一致;唯一版本差异 bddl
(本机 3.6.0 被 omnigibson 带入 vs ARM 1.0.1)已实测排除:PYTHONPATH
前置 bddl 1.0.1(`$OFT_OUTPUT/libero-runtime/`)重跑 16 条对照结果不变
(68.75%,失败任务同为 {1,6,8,11,12})。fp32 对照 75%(12/16),失败
集合微移,精度只是次要因素。

结论:两机在软件可对齐的范围内已经一致,剩余差距来自**平台级二进制
差异**——同版本 MuJoCo 的 x86_64 与 aarch64 build 在接触物理与渲染上
的浮点行为不同,叠加不同 GPU/EGL 驱动的图像渲染差异;512 步接触密集
操作会把微小差异放大成确定性的按任务成败翻转。这类差异无法跨平台对齐。

2026-08-14 追加两项定界(均按外部 review 意见执行):

7. 评错权重:排除。三个 GRPO 诊断 run 的 resolved config
   `rollout.model.model_path` 均指向
   `/workspace/output/RLinf-OpenVLAOFT-GRPO-LIBERO-90/`(分片 sha256 与
   HF 一致),base run 指向 Base-Lora;没有把 GSE 配置默认的 SFT 路径误当
   GRPO 评。ARM 侧同协议参考值修正为:GRPO 96.289%、base 81.836%、
   16 条对照 GRPO 100% / base 87.5%。
8. venv.py respawn 补丁(12.5a):排除。用 stock venv.py(git checkout
   还原,评估含初始 reconfigure 路径)重跑 16 条对照,结果与补丁版
   **逐位一致**(68.75%,失败任务同为 {1,6,8,11,12})。补丁只解决主机
   内存泄漏,不改变评估行为;本机评估完全确定性,失败指纹已三次复现。

由于官方也是在 x86 平台评得 96.44%,"CPU 架构差异"不能解释全部;
三方(官方 x86 / ARM 集群 / 本机 x86)真正可能不同、且尚未排除的是:

- **GPU/EGL 驱动渲染差异**(A100-PCIE vs A100-SXM、不同 driver 的
  光栅化/光照细节)。本机 driver `535.183.06`。
- 官方口径为 90 任务全 50 trial(4500 条),trial 子集差异可解释数个
  百分点,不足以解释 20 pp。

2026-08-14 继续排除(累计第 9–10 项):

9. Pillow 版本(本机 12.2.0 vs ARM 11.0.0):排除。venv 安装 11.0.0 后
   16 条对照仍逐位一致(68.75%,失败集 {1,6,8,11,12} 第四次复现)。
   顺带说明观测 resize 走 tensor 路径,不经 PIL。已保留 11.0.0。
10. transformers 来源:一致。本机 venv 的 4.40.1 就是 moojink
    `transformers-openvla-oft` fork(`direct_url.json` commit
    `bc339d9`),与 ARM compat 目录的 fork 同源;timm 0.9.10、
    tokenizers 0.19.1、numpy 1.26.4 双机一致。仅 torchvision
    (0.21.0 vs NGC 0.20.0a0)与 opencv(4.11 vs 5.0,均不在观测路径)
    存在版本差。

**跨机观测/动作探针**已就绪:
[scripts/probe_libero_obs.py](scripts/probe_libero_obs.py)。它渲染
LIBERO-90 固定 init state(默认 task 1 trial 0,含 15 步 settle),保存
`obs.npy/obs.png` 并打印图像 sha256 与统计,再用 GRPO 权重跑一次 greedy
输出动作向量;支持 `--npy` 喂入对方机器的观测以隔离模型侧。本机(x86)
基准:image sha256 `f2166cf2775f1977`、mean 110.7331、std 72.9005,
首 chunk greedy 动作已存 `$OFT_OUTPUT/probe_x86/`。

探针跨机结果(2026-08-14):**ARM 渲染与本机基本无差异**——观测/渲染
差异假设排除。剩余差距的候选收敛为:模型前向数值(两机 greedy 动作向量
对比进行中)与 512 步闭环中的物理演化差异。两机绝对成功率仍不可混报。

纪律不变:两机绝对成功率**不可混合比较**;同机内 base→GSE→GRPO 的
相对比较有效(本机口径:base 55.9% < GSE step-30 61.5% < 官方 GRPO
75.0%)。

独立 greedy 评估任意权重的命令模板(在容器内、12.3 初始化后):

```bash
export EVAL_DIR=/workspace/output/<eval_name> && mkdir -p $EVAL_DIR
python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name libero_90_grpo_openvlaoft_gse_r32_svd \
  '+model@rollout.model=openvla_oft' \
  rollout.model.model_path=<HF 布局权重目录> \
  rollout.model.unnorm_key=libero_90_no_noops_trajall \
  rollout.model.max_prompt_length=128 \
  rollout.model.attn_implementation=sdpa \
  actor.model.gse.enabled=False \
  runner.only_eval=True \
  env.eval.total_num_envs=64 env.eval.rollout_epoch=8 \
  rollout.micro_batch_size=8 \
  runner.logger.log_path="$EVAL_DIR" \
  runner.logger.experiment_name=<eval_name> \
  2>&1 | tee $EVAL_DIR/console.log
```

要点:`only_eval` 模式下 worker 从 `cfg.rollout.model` 建模,必须用
`'+model@rollout.model=openvla_oft'` 合并模型组并覆盖
`unnorm_key`/`max_prompt_length`(组默认是 LIBERO-10 的值);评估 GSE
checkpoint 时再加 `rollout.model.gse.*` 全套字段和
`runner.ckpt_path=<ckpt>/actor/model_state_dict/full_weights.pt`。评估
temperature 语义:`temperature_eval > 0` 即随机采样,greedy 必须用 `-1`
(官方 `evaluations/libero/*_eval.yaml` 里 `do_sample: False` 配
`temperature_eval: 1.6` 实际会走随机采样,并行文档 13.2 的 2.7% 之谜
即源于此)。

### 12.5c W&B 记录与自动重评说明(2026-08-13)

- 循环脚本内置 `WANDB_OVERRIDES` 构造:shell 数组不会传入子进程,最初在
  交互 shell 里定义数组再调脚本导致整段训练只有 TensorBoard 记录。现在只需
  容器环境里有 `WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY`,脚本自动启用
  W&B,缺任一则自动回退 tensorboard-only。
- 让续跑重做某个 checkpoint 的评估:删除该
  `global_step_<N>/evaluation_complete.json`(容器内 root 权限)后以
  `+runner.eval_on_resume=true`(脚本已带)续跑即可;flash-attn 时代的
  step-10/20 评估数值(3.4–3.6%)全部无效,不得用于曲线。

### 12.5d 官方评估协议与 whole-model GSE 正式训练(2026-08-14)

本小节记录已废止的视觉 GSE 实验。其 437 层、参数量、显存、吞吐和 checkpoint
不能代表 2026-08-21 起的 frozen-vision PE-RL，也不得作为新方法的恢复点。

最终同机、同权重、同 fixed-reset 窗口对照定位出此前 20--25 pp 差距的
核心原因，不是 CPU/GPU 仿真平台差异，而是项目评估路径同时偏离官方协议:

1. 环境观测缺少 OpenVLA-OFT 官方图像变换：JPEG round-trip、Lanczos
   resize 到 224，以及 0.9 center crop。现在由
   `env/libero_90.yaml: official_image_preprocess: true` 固化。
2. 周期评估曾把 `temperature_eval` 设为 `-1`，实际走 greedy。官方发布
   模型的对照协议是 `temperature_eval: 1.6`、`top_k: -1`、`top_p: 1.0`。
3. 注意力实现必须为 SDPA；FA2 的行为级损坏是独立问题，不能启用。

修复后三组 256 条 fixed-window 结果如下，均来自本八卡服务器:

| 模型 | 图像/采样/attention | `eval/success_once` |
|---|---|---:|
| Base-SFT-Lora | 官方图像 + temperature 1.6 + SDPA | 220/256 = **85.9375%** |
| 官方 GRPO | 官方图像 + temperature 1.6 + SDPA | 248/256 = **96.875%** |
| 官方 GRPO | 原始图像 + greedy + SDPA（旧错误协议） | 约 71.5--75.0% |

96.875% 与官方/ARM 的约 96% 对齐，证明本机 CPU 物理 + GPU EGL 渲染
没有导致先前宣称的系统性成功率缺口。论文只允许比较同一图像预处理、
采样温度、attention 实现、reset 窗口与轨迹数下的结果。

为使方法定义严格，正式配置不再只向 LLM 注入 GSE 并全量训练视觉/OFT
权重，而是使用:

```yaml
actor:
  model:
    attn_implementation: sdpa
    gse:
      scope: whole_model
      target_modules: all-linear
      total_rank: 32
      num_experts: 8
      num_generalized_experts: 1
      top_k: 2
      initialization: svd
      freeze_base: true
```

对正式 Base checkpoint 的实际枚举为 437 个 `torch.nn.Linear`：视觉骨干
209 层、multimodal projector 3 层、语言模型 225 层；最小输入/输出维度
为 1024，满足 rank-32 SVD 约束。所有 dense 原始参数冻结，只有这 437 层
的 GSE experts/router 可训练，共 121,558,592 个参数；真实 7B 模型预检
确认没有任何非 adapter 参数保持 `requires_grad=True`。旧目录
`libero90_gse_r32_svd_8gpu_seed1234` 属于 LLM-only/非最终评估协议诊断，
不得续接或用于论文。

正式运行身份与路径:

```text
experiment: libero90_gse_whole_model_r32_svd_8gpu_seed1234
RUN_DIR: /workspace/output/libero90_gse_whole_model_r32_svd_8gpu_seed1234
launcher: scripts/run_libero90_gse_8gpu_loop.sh
tmux: rlinf-train:0
```

自动循环只会在上述新目录中寻找完整 checkpoint，因此首次启动必定从
Base step 0 开始；后续异常退出才允许在完全相同协议下自动恢复。实际
attempt 1 于 2026-08-14 14:05 UTC（22:05 CST）以 `resume_dir=null` 启动，
W&B run id 为 `dkj7waye`。rollout/actor 两组均在日志中确认注入 437 层。

后续确认 attempt 1--3 均不健康：step 0 rollout 完成后的首次 actor
backward 都以相同错误退出，且未产生训练指标或 checkpoint：

```text
RuntimeError: setStorage: sizes [7, 4304], storage offset 174592, ...
are out of bounds for storage of size 0
```

`[7, 4304]` 正是视觉塔 GSE router（7 个 specialized experts）的权重
形状。逐项关闭 reentrant checkpointing、全部 gradient checkpointing、
actor offload，以及启用 `use_orig_params=True` 后错误均原样复现，排除了
这些配置项。真实根因是 `GSEAdapter` 只把 residual 作为 FSDP forward
输出，却把可微的 router load-balancing loss 保存在模块属性中；worker
随后在 FSDP 边界外把该 loss（即使系数为 0）加入总 loss。FSDP 看不到
这条输出支路，因而未注册 pre-backward unshard hook，反向直接访问了
reshard 后的 0-storage router view。

修复后 `GSEAdapter.forward()` 同时返回 residual 和 load-balancing loss，
由 `GSELinear` 解包保存；`GSELinear` 的外部接口仍只返回普通 tensor。
这样 auxiliary 支路成为 FSDP 正式输出并获得 unshard hook。2026-08-15
的 8 卡最小合法 GRPO 验收（16 env、group size 2、1 rollout epoch、
64 steps、whole-model 437 层、原始 full-shard/offload/reentrant 设置）已
完整完成 `Global Step 1/1`：`actor/run_training=45.944 s`、
`Step Time=103.860 s`，optimizer step 与 LLM/vision/projector router
指标全部产生，未再出现 storage 错误。正式配置把两个 auxiliary-loss
系数显式固定为 0.0；它们仅作诊断，不参与论文主实验优化目标。旧正式
输出只作失败诊断，禁止续接。

修复后的正式 attempt 1 于 2026-08-15 00:32 CST（2026-08-14 16:32 UTC）
从 `resume_dir=null` 重启，W&B run id 为 `kuz5sy2b`。旧目录可恢复地归档为
`/workspace/output/libero90_gse_whole_model_r32_svd_8gpu_seed1234_failed_fsdp_aux_20260815`
（43 MiB）；新的 canonical `RUN_DIR` 未混入旧 TensorBoard/W&B 事件。正式
step 0 已完整落盘并继续进入 step 1 rollout：256 trajectories、
`success_once=77.734375%`、`return=3.88671875`、`approx_kl=0.0020913`、
`grad_norm=0.390359`、actor update 396.259 s、整步 1107.184 s。router
active layers 为 vision 204、projector 3、LLM 225；load-balancing loss
记录为 1.23843，但其 weighted value 为 0，符合正式配置。该 step 在原始
full-shard、actor offload、reentrant checkpointing、micro batch 32 和
global batch 1024 下完成，是正式规模的修复验收，不是缩容冒烟结果。

### 12.6 验收与回退规则

- 首次 step-10 评估必须核对:`eval/num_trajectories=512`,且逐任务字段覆盖
  全部 90 个 LIBERO-90 任务。`64×8` 与四卡 `32×16` 总量一致且同走多轮
  auto-reset 推进路径;若触发显存/主机内存/framebuffer 故障,回退四卡已
  验证的 `32×16`(总量仍为 512)。不要尝试 `512×1` 或任何 >16 env/GPU 的
  单轮形态,EGL 会在初始化阶段直接失败。**本机已实测通过**(2026-08-13,
  重生补丁后的 attempt-1 run):512 trajectories、`covered_tasks=90`、
  90 个逐任务字段齐全,checkpoint 的 DCP `.metadata` 与 `full_weights.pt`
  完整,`best_macro_mean` 快照生成;含保存+评估的 step 耗时约 1519 s
  (纯训练 step 约 727 s)。
- 评估健康基线以 12.5d 的官方协议为准：Base-SFT 85.9375%、官方 GRPO
  96.875%（各 256 条）。55.9%/71.5--75.0% 是旧图像/greedy 错误协议，
  flash-attn 时代的 3.4--4.5% 同样无效，全部不得进入论文曲线。
- 当前官方图像/temperature-1.6 协议从成功率较高的 Base-SFT 初始化，正式
  step 0 不应再全零。本次 256 条得到 `success_once=77.734375%`、
  `grad_norm=0.390359`；若 step 0 全零，应立即检查图像预处理、采样温度、
  SDPA 和 reset 窗口。旧协议下“step 0 全零、step 1 为 3.125%”的记录不再
  是当前实验的健康基线。
- whole-model GSE 当前吞吐基线（正式 step 0）：整步 1107.184 s，其中
  rollout 674.054 s（两个 epoch 的生成主体 648.525 s）、actor update
  396.259 s、权重同步 36.800 s。旧 LLM-only 配置的 695--713 s/step 不可
  用于当前回归。后续稳态可因环境成功提前终止而更快，但若显著变慢，应先
  核对 `rollout.micro_batch_size=8` 与 `actor.micro_batch_size=32`。
- whole-model 正式 backward 峰值约 57 GiB/卡，80 GiB A100 仍有约 23 GiB
  余量。不要再沿用 LLM-only 的 39 GiB 峰值。actor 若在其他改动后 OOM，
  回退 `actor.micro_batch_size=16`，且续跑必须保持相同值；变更模型、指标
  或 batch 协议后必须重新校准。
- 主机内存(1 TB)远高于并行服务器的 254 GiB 上限,四卡文档中的
  Ray CPU/内存阈值限制不适用于本机;沿用第 7 节的 Ray scratch 重定向即可,
  但仍应在首个 step 监控 `free -g` 与 `du -sh $RAY_SCRATCH`。
- 训练数值以 `RUN_DIR` 下的 `metrics.jsonl`、TensorBoard 与 W&B 为准;
  checkpoint 位于 `RUN_DIR/<experiment_name>/checkpoints/global_step_<N>/`,
  最佳权重快照位于 `checkpoints/best_macro_mean/`。

### 12.7 Frozen-vision PE-RL 八卡协议(2026-08-21，当前有效)

当前方法把视觉 backbone 定义为冻结状态编码器。配置必须同时满足：

```yaml
actor:
  model:
    gse:
      scope: whole_model
      target_modules: all-linear
      freeze_vision_backbone: true
      semantic_conditioning: true
      routing_granularity: token
      action_sequence_routing: true
env:
  train:
    official_image_preprocess: false
  eval:
    official_image_preprocess: false
```

`all-linear` 在该组合下只枚举 projector 和 language model；不会向
`vision_backbone` 注入 GSE。初始化完成后必须验证：

```text
all(not p.requires_grad for p in model.vision_backbone.parameters()) == True
all(not name.startswith("vision_backbone")
    for name in model.gse_injection_report.injected_module_names) == True
```

projector 与普通 LLM token 采用 token-level 路由；动作 token 位置采用共享的
sequence-level 当前状态。所有 router 同时接收从 instruction token 提取的冻结
文本 embedding，不使用 task ID。语义 projection 属于 GSE adapter，必须出现在
optimizer 和 checkpoint 中。

八卡启动入口仍为：

```bash
export RUN_DIR=/workspace/output/libero90_gse_frozen_vision_r32_svd_8gpu_seed1234
mkdir -p "$RUN_DIR"
bash scripts/run_libero90_gse_8gpu_loop.sh
```

脚本默认 `actor.global_batch_size=16384`、actor micro batch 32、rollout micro
batch 8、gradient checkpointing 开启，并启用 W&B（环境凭据完整时）。不得将
`INITIAL_RESUME` 或 `runner.resume_dir` 指向任何
`libero90_gse_whole_model_*` checkpoint：视觉 GSE 已被移除且新增 semantic
router，模型和 optimizer state 均不兼容；必须从 Base-SFT 启动新实验。

当前主配置按实验要求在 train/eval 同时关闭 `official_image_preprocess`，使用
一致的 raw-image 输入。12.5d 的 Base-SFT 85.9375% 和官方 GRPO 96.875%
来自官方图像预处理，只能验证旧官方协议，不能作为本次 raw-image 训练的健康
阈值。比较 Base-SFT、full GRPO 与 PE-RL 时必须全部使用相同的 raw-image
配置；不得跨图像协议比较绝对成功率。

12.5d 的 437 层、121,558,592 个 adapter 参数、57 GiB backward 峰值和
1107.184 s/step 都是旧 whole-model 实测值。冻结视觉后的注入层数、可训练参数、
显存和吞吐尚未完成真实 7B 八卡测量，首个正式 step 必须重新记录，不能沿用旧值。
