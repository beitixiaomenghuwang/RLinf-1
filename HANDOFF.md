# Pi0.5 Multi-Task Parameter-Efficient RL Handoff

Last updated: 2026-07-28

This document contains only the current method, validated results, operational
constraints, and next experiments. Historical pilots and resolved failures have
been removed; use Git history when those details are needed.

## 1. Current status

The project studies parameter-efficient multi-task RL post-training of one
Pi0.5 policy on MetaWorld MT50. The RL algorithm and critic are intentionally
kept identical to RLinf: all current runs use Flow-SDE PPO with GAE. The research
variable is the trainable policy parameterization.

The current main method is action-head GSE:

1. Load the fully trained Pi0.5 MT50 SFT checkpoint.
2. Freeze the pretrained VLM and action-expert weights.
3. Add zero-output low-rank residual experts to the action transformer.
4. Train only GSE, its router, and the value head with standard RLinf PPO.

The selected action-GSE checkpoint is training step 180. It is the highest
validation checkpoint in the converged step-20-to-step-220 region, rather than an
early-training pilot. Three Flow-SDE rollout seeds evaluated from this same
training checkpoint give `72.72%` mean success-once and `72.73%` macro success.

The main numerical statement is currently:

> Action-GSE averages 72.72% success-once, 2.02 percentage points above the
> 70.7% result reported by RLinf, while training about 30.15M parameters of a
> 3.65B-parameter policy.

This is not yet a strict claim that GSE outperforms the released full-parameter
checkpoint. A local 512-trajectory evaluation of that checkpoint obtained
`74.22%` from one rollout seed. The released checkpoint and GSE must be evaluated
over the same multiple seeds and reset states before the final paper claim.

## 2. Repositories and assets

```text
Development repository: /home/caslx/Robotics/RLinf
Reference repository:   /home/caslx/Robotics/VLA-GSE
Docker image:           rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
```

Server paths currently in use:

```bash
export RLINF_REPO=/home/xueyang/RLinf
export PI05_SFT=/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-SFT
export PI05_RL=/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-RL-FlowSDE
export GSE_OUTPUT=/DATA/disk0/xueyang/model/pi05-gse
export HF_CACHE=/home/xueyang/RLinf/cache/huggingface
export RAY_SCRATCH=/DATA/disk0/xueyang/Data/rlinf-ray
```

Inside Docker these are mounted as:

```text
/workspace/RLinf
/workspace/models/RLinf-Pi05-MetaWorld-SFT
/workspace/models/RLinf-Pi05-MetaWorld-RL-FlowSDE
/workspace/output
/workspace/ray
```

Activate the model environment with `source switch_env openpi` and set
`EMBODIED_PATH=/workspace/RLinf/examples/embodiment`.

## 3. Implemented methods

### 3.1 Action-GSE

Primary configuration:

```text
examples/embodiment/config/metaworld_50_ppo_openpi_pi05_gse.yaml
```

Architecture per wrapped linear layer:

- total rank 64 and alpha 64;
- eight experts: two always-active generalized and six routed specialized;
- specialized `top_k=2` with sequence-mean routing;
- joint orthogonal A initialization and zero B initialization;
- targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj` in all 18 action-transformer layers, 126 wrapped projections total.

The base model is unchanged at initialization because every B matrix is zero.
The first backward pass trains B; A and router receive useful policy gradients
after B becomes nonzero. Do not describe this as all-zero LoRA initialization.

Model size measured on the real checkpoint:

```text
total policy parameters:                  3,646,909,201
GSE adapter parameters:                      28,938,240
trainable parameters including value head:   30,151,681
```

Load-balance and orthogonality losses are implemented but disabled in the main
run (`load_balancing_loss_coef=0`, `orthogonality_loss_coef=0`). They are logged
as diagnostics and do not modify PPO.

### 3.2 Action-head plain LoRA

Parameter-matched baseline configuration:

```text
examples/embodiment/config/metaworld_50_ppo_openpi_pi05_action_lora.yaml
```

Plain LoRA uses rank 64, alpha 64, PEFT Gaussian A and zero B, and exactly the
same 126 action-transformer projections as GSE. The base policy is frozen and
only LoRA plus the value head train. Full-state checkpoint detection and loading
are supported by:

```text
rlinf/models/embodiment/openpi/lora.py
```

This dedicated path is required because RLinf's older generic OpenPI LoRA path
targets the VLM and is not an action-head-matched baseline.

### 3.3 VLM extension

The isolated VLM experiment loads action-GSE step 180, freezes all 126 trained
action adapters, and adds zero-output GSE to VLM language layers 14-17:

```text
examples/embodiment/config/metaworld_50_ppo_openpi_pi05_gse_vlm_last4.yaml
```

It wraps 28 VLM projections with total rank 64, four experts, one generalized
expert, and specialized `top_k=2`. The VLM-GSE portion has 17,776,640 parameters;
including the value head, about 19M parameters train. No matched result is
recorded in this handoff yet.

## 4. Results

### 4.1 Aggregate comparison

The protocols are shown explicitly because the current evaluations are not all
matched.

| Method | Checkpoint/eval | Success once | Macro | End | Worst-10 | >=90 tasks |
|---|---|---:|---:|---:|---:|---:|
| RLinf reported Flow-SDE | paper result | 70.70% | unavailable | unavailable | unavailable | unavailable |
| Released full Flow-SDE | 1 rollout seed, 512 trajectories | 74.22% | 74.20% | 42.58% | 16.20% | 18 |
| Action-GSE | step 180, 3 rollout seeds, 512 each | 72.72% | 72.73% | 47.20% | 18.73% | 21.0 |
| Action-GSE | step 180, best rollout seed 42 | about 75% | about 75% | see raw metrics | see raw metrics | see raw metrics |
| Plain LoRA | step 80, 1 seed, 448 trajectories | 63.17% | 63.22% | 42.86% | 4.44% | 19 |

The action-GSE three-seed summary is:

```text
success_once: mean 72.72%, std 2.16%, approximate 95% CI [70.28%, 75.17%]
macro mean:   mean 72.73%, std 2.07%, approximate 95% CI [70.38%, 75.07%]
success_end:  mean 47.20%, std 1.24%
worst-10:     mean 18.73%, std 6.32%
worst-5:      mean  6.91%, std 6.50%
tasks >=90%: mean 21.0
```

These are three stochastic evaluation seeds from one training seed, not three
independent training runs. The confidence intervals therefore quantify rollout
variation only.

### 4.2 Plain-LoRA training curve

The seven-GPU run collects 224 trajectories per global step and evaluates with
448 fixed-reset trajectories. The best macro checkpoint through step 160 is
step 80:

| Step | Success once | Macro | End | Worst-10 | >=90 tasks |
|---:|---:|---:|---:|---:|---:|
| 20 | 61.38% | 61.42% | 44.20% | 4.44% | 17 |
| 80 | **63.17%** | **63.22%** | 42.86% | 4.44% | 19 |
| 160 | 62.50% | 62.67% | 40.63% | **7.78%** | 17 |

Intermediate checkpoints remain between 59.11% and 61.53% macro success. PPO
training is numerically healthy: observed approximate KL stays below `0.008`,
clip fraction below `0.108`, gradients are finite, and critic explained variance
is approximately `0.66-0.77`.

The current evidence strongly rejects the hypothesis that equal total low-rank
capacity alone explains GSE: action-GSE exceeds the best plain-LoRA macro result
by about `9.50 pp` and worst-10 by about `14.28 pp`. This is not yet a formal
matched comparison because GPU count, evaluation trajectory count, training
environment steps, and evaluation seeds differ.

### 4.3 Interpretation for the paper

Supported now:

- parameter-efficient RL post-training substantially improves the SFT policy;
- the routed generalized/specialized parameterization is much stronger than the
  current parameter-matched plain-LoRA run;
- action-GSE is slightly above RLinf's reported 70.7% average and improves tail
  metrics relative to the locally evaluated released checkpoint.

Not supported yet:

- statistically significant superiority over the released full-parameter model;
- task-specific expert specialization;
- improved out-of-distribution generalization;
- conclusions from independent training seeds.

At action-GSE step 180, normalized router entropy is `0.954`, task-router NMI is
`5.63e-4`, and task-router JS divergence is `2.53e-4`. The router is not globally
collapsed but is effectively task-agnostic. The current gain is best described
as conditional/shared residual capacity, not proven per-task specialization.

## 5. Evaluation and checkpoint rules

Eval-only workers load only `rollout.model.model_path`; they do not synchronize
weights from the actor. For a saved RL checkpoint, point this path to its `actor`
directory containing `model_state_dict/full_weights.pt`. Keep the actor model
path valid, but do not use `runner.resume_dir` for standalone evaluation.

For GSE or action-LoRA checkpoints, mirror the complete adapter configuration
under `rollout.model` so the wrapped structure exists before full weights load.
For the released full Flow-SDE checkpoint, use the base Pi0.5 config with
`rollout.model.openpi.noise_method=flow_sde` and no adapters.

Do not compare methods using different final protocols. The next formal
evaluation should use:

- the same fixed reset-state IDs and Flow-SDE seeds for all methods;
- either 512 trajectories on eight GPUs for every method, or a new count shared
  by every method;
- a validation reset pool for checkpoint selection and a separate final pool;
- at least three rollout seeds per selected checkpoint;
- at least three independent training seeds for the final methods.

When only GPU ranks 1-7 are allowed but the connected Ray cluster advertises all
eight GPUs, add:

```bash
'cluster.component_placement={actor\,env:1-7,rollout:1-7}'
```

For seven actor ranks with micro batch 128, `actor.global_batch_size` must be a
multiple of `896`. The validated plain-LoRA setup uses 224 train environments,
one rollout epoch, global batch 896, and 448 eval environments.

## 6. Immediate next experiments

Priority order:

1. Re-evaluate the released full Flow-SDE checkpoint with the same three rollout
   seeds and fixed 512 reset states used for action-GSE.
2. Re-evaluate action-GSE step 180 and plain-LoRA step 80 under one identical
   protocol. Continue plain LoRA only if its training budget is still materially
   below action-GSE.
3. Run at least three independent training seeds for full RL, plain LoRA, and
   action-GSE after the protocol is frozen.
4. Evaluate VLM-last4 against two equal-budget controls: frozen action-GSE with
   VLM-GSE, and continued action-only GSE training for the same extra RL steps.
5. Add the no-router multi-expert residual baseline to isolate routing from
   orthogonal multi-expert capacity.
6. Run initialization ablations: orthogonal-zero, scale-matched
   orthogonal-zero, and Kaiming/Gaussian zero-output initialization.
7. Only after matched in-distribution results, test held-out reset states and
   visual/state perturbations for a generalization claim.

Primary paper metrics are macro success, success-at-end, median, worst-5,
worst-10, tasks above 90%, per-task gains/regressions from SFT, success-versus-env
steps AUC, trainable and active parameters, optimizer memory, peak GPU memory,
wall time, and inference latency.

## 7. Stable implementation constraints

- Main training uses Flow-SDE from `model/pi0_5.yaml`; it is not Flow-Noise.
- GSE and plain LoRA are mutually exclusive.
- GSE FSDP uses `no_shard` and `use_orig_params: False`; do not change this
  without a smoke test.
- Keep auxiliary load-balance and orthogonality coefficients at zero unless a
  dedicated ablation justifies changing the PPO objective.
- Do not infer task specialization from aggregate entropy or uniform usage.
- Preserve checkpoints and large logs outside the repository.
- Do not terminate unrelated GPU jobs or remove shared Ray/Docker data.

Focused unit tests:

```bash
pytest -q \
  tests/unit_tests/models/test_gse.py \
  tests/unit_tests/models/test_openpi_gse.py \
  tests/unit_tests/models/test_openpi_action_lora.py \
  tests/unit_tests/test_task_advantage_metrics.py
```

Important recent commits:

```text
c798a718  reproducible rollout seeds and multi-seed summaries
8c59a4a9  load action-GSE checkpoints before optional VLM injection
6e4fa76c  distributed per-task advantage metric fix
e89968d2  VLM-last-four GSE configuration
51fe065d  action-head plain-LoRA baseline and full-checkpoint loading
```

After completing a code, configuration, test, or documentation stage, inspect
the diff, run focused validation, and create a signed Conventional Commit. Do not
stage checkpoints, logs, or unrelated user changes.
