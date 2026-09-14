#!/bin/bash
# Container-side training loop for the MetaWorld MT50 rank-64 NO-ROUTER
# multi-expert arm (PAPER_SUMMARY.md 4.4 ablation A) on 8 x A100-SXM4-80GB.
#
# Run this INSIDE the OpenPI container after `source switch_env openpi`, with
# RUN_DIR / EXP_NAME / EMBODIED_PATH exported (HANDOFF.md section 7).
# scripts/launch_metaworld_gse_r64_no_router_8gpu.sh does all of that for you.
#
# The capacity numbers below are the ones measured on THIS machine for the
# rank-64 2g6s run gse-formal-seed42 (HANDOFF.md 9.1/9.2): 128 train envs x 2
# rollout epochs = 256 trajectories/step = 5120 PPO chunk samples/step, actor
# micro batch 128 x 8 ranks = global batch 1024 (one accumulation step), and a
# 512-trajectory fixed-reset evaluation. They are NOT the BSCC 8-GPU values
# (micro batch 16, eval 32x16) -- that cluster has a tighter host-memory cap.
# The no-router arm has the same footprint as the routed one: both routing
# paths call _fused_expert_residual over ALL experts, so the fused rank_hidden
# is total_rank wide either way and the only thing dropped here is the router
# logits.
set -u

: "${RUN_DIR:?export RUN_DIR (e.g. /workspace/output/<experiment>)}"
: "${EXP_NAME:?export EXP_NAME}"
EMBODIED_PATH=${EMBODIED_PATH:-/workspace/RLinf/examples/embodiment}

CONFIG=${CONFIG:-metaworld_50_ppo_openpi_pi05_gse_action_r64_svd_no_router}
SFT_MODEL=${SFT_MODEL:-/workspace/models/RLinf-Pi05-MetaWorld-SFT}

# Capacity. Override only with a reason, and record it in HANDOFF.md.
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-128}
ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE:-1024}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-8}
TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS:-128}
TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-2}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-512}
EVAL_ROLLOUT_EPOCH=${EVAL_ROLLOUT_EPOCH:-1}

SEED=${SEED:-42}
# Empty means "take the value the YAML pins", which is the matched setting.
# GSE_LORA_ALPHA=128.0 builds the SEPARATE gate-mass-matched companion arm
# described in the config header; it is not a substitute for this one.
GSE_LORA_ALPHA=${GSE_LORA_ALPHA:-}
ACTOR_LR=${ACTOR_LR:-}

MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
RETRY_SLEEP=${RETRY_SLEEP:-30}
# PREFLIGHT_ONLY=1 resolves and gates the config, then exits without training.
# Safe to run while the GPUs belong to someone else: `--cfg job` short-circuits
# inside Hydra before main() runs, so no Cluster, no Ray and no CUDA context.
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
CKPT_ROOT="$RUN_DIR/$EXP_NAME/checkpoints"

mkdir -p "$RUN_DIR"
test -f "$SFT_MODEL/model.safetensors" || {
  echo "[preflight] missing SFT weights at $SFT_MODEL/model.safetensors" >&2
  exit 3
}

# Shell arrays are not inherited by child processes, so the W&B overrides have
# to be built here and not in the interactive shell -- a LIBERO run on
# 2026-08-13 lost all W&B logging to exactly that mistake. A mounted ~/.netrc
# is accepted so WANDB_API_KEY need not sit in the process environment.
if { [ -n "${WANDB_API_KEY:-}" ] || [ -s "${HOME}/.netrc" ]; } &&
  [ -n "${WANDB_PROJECT:-}" ] && [ -n "${WANDB_ENTITY:-}" ]; then
  WANDB_OVERRIDES=(
    'runner.logger.logger_backends=[tensorboard,wandb]'
    "runner.logger.project_name=${WANDB_PROJECT}"
    "+runner.logger.wandb_entity=${WANDB_ENTITY}"
  )
else
  echo "[loop] WANDB_* incomplete; logging to tensorboard only" >&2
  WANDB_OVERRIDES=('runner.logger.logger_backends=[tensorboard]')
fi

# Every key named here already exists in the config chain, so none of them may
# carry a "+" prefix; Hydra fails composition with "Could not append to config"
# when "+" is used on an existing key. runner.eval_on_resume is the exception:
# the embodied runner reads it with .get(), it is absent from these YAMLs, and
# Hydra runs in struct mode, so it must be appended.
BASE_OVERRIDES=(
  "runner.logger.log_path=$RUN_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
  "actor.model.model_path=$SFT_MODEL"
  "rollout.model.model_path=$SFT_MODEL"
  "env.train.total_num_envs=$TRAIN_NUM_ENVS"
  "env.train.rollout_epoch=$TRAIN_ROLLOUT_EPOCH"
  "env.eval.total_num_envs=$EVAL_NUM_ENVS"
  "env.eval.rollout_epoch=$EVAL_ROLLOUT_EPOCH"
  "actor.micro_batch_size=$ACTOR_MICRO_BATCH_SIZE"
  "actor.global_batch_size=$ACTOR_GLOBAL_BATCH_SIZE"
  "rollout.micro_batch_size=$ROLLOUT_MICRO_BATCH_SIZE"
  "actor.seed=$SEED"
  "rollout.seed=$SEED"
)
if [ -n "$GSE_LORA_ALPHA" ]; then
  BASE_OVERRIDES+=("actor.model.gse.lora_alpha=$GSE_LORA_ALPHA")
  echo "[loop] lora_alpha overridden to $GSE_LORA_ALPHA (separate arm, not the matched one)"
fi
if [ -n "$ACTOR_LR" ]; then
  BASE_OVERRIDES+=("actor.optim.lr=$ACTOR_LR")
  echo "[loop] actor lr overridden to $ACTOR_LR (no longer matched to the main method)"
fi

# --- config gate -------------------------------------------------------------
# Read the RESOLVED config, not the edited YAML: launcher CLI overrides outrank
# the YAML, and a resumed scheduler can outrank actor.optim.lr. `--cfg job`
# short-circuits inside Hydra before main() runs, so validate_cfg is never
# called and no Cluster/Ray is started by this step.
RESOLVED="$RUN_DIR/resolved_config.yaml"
if ! python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name "$CONFIG" \
  --cfg job --resolve \
  "${BASE_OVERRIDES[@]}" >"$RESOLVED" 2>"$RUN_DIR/resolve_config.err"; then
  echo "[preflight] Hydra could not compose $CONFIG; see $RUN_DIR/resolve_config.err" >&2
  tail -20 "$RUN_DIR/resolve_config.err" >&2
  exit 4
fi

python - "$RESOLVED" "$SFT_MODEL" <<'PY' || exit 4
import sys

import yaml

resolved_path, sft_model = sys.argv[1], sys.argv[2]
with open(resolved_path) as handle:
    cfg = yaml.safe_load(handle)

runner, actor, env = cfg["runner"], cfg["actor"], cfg["env"]
gse, optim, fsdp = actor["model"]["gse"], actor["optim"], actor["fsdp_config"]
failures = []


def expect(label, actual, wanted):
    if actual != wanted:
        failures.append(f"{label}: expected {wanted!r}, resolved {actual!r}")


# No router at all: num_specialized_experts == 0 makes GSEAdapter build an
# nn.Identity() in place of the router Linear, so nothing routes and nothing
# router-shaped is trainable.
expect("gse.enabled", gse["enabled"], True)
expect("gse.routing_mode", gse["routing_mode"], "uniform")
expect("gse.num_experts", gse["num_experts"], 8)
expect("gse.num_generalized_experts", gse["num_generalized_experts"], 8)

# Matched against the rank-64 main method.
expect("gse.total_rank", gse["total_rank"], 64)
expect("gse.scaling_mode", gse["scaling_mode"], "total_rank")
expect("gse.initialization", gse["initialization"], "orthogonal_zero")
expect("gse.lora_dropout", float(gse["lora_dropout"]), 0.0)
expect("gse.load_balancing_loss_coef", float(gse["load_balancing_loss_coef"]), 0.0)
expect("gse.orthogonality_loss_coef", float(gse["orthogonality_loss_coef"]), 0.0)
expect("gse.semantic_conditioning", gse["semantic_conditioning"], False)
expect("gse.train_action_adapters", gse["train_action_adapters"], True)
expect("gse.init_seed", gse["init_seed"], 42)
if "gse_router_lr" in optim:
    failures.append(
        "actor.optim.gse_router_lr is set; the main method shares the actor lr "
        "and this arm has no router to give a separate lr to"
    )

# Budget and cadence.
expect("runner.max_epochs", runner["max_epochs"], 300)
expect("runner.save_interval", runner["save_interval"], 10)
expect("runner.val_check_interval", runner["val_check_interval"], 20)
expect("runner.max_checkpoints_to_keep", runner["max_checkpoints_to_keep"], None)
expect("runner.only_eval", runner["only_eval"], False)
expect("optim.total_training_steps", optim["total_training_steps"], runner["max_epochs"])
expect("optim.lr_scheduler", optim["lr_scheduler"], "cosine")
expect("optim.min_lr_rate", float(optim["min_lr_rate"]), 0.1)
expect("optim.lr_warmup_steps", optim["lr_warmup_steps"], 0)

# Adapter-only checkpoints, no full-weight dump.
expect("fsdp.adapter_only_checkpoint", fsdp["adapter_only_checkpoint"], True)
expect("fsdp.save_full_model_weights", fsdp["save_full_model_weights"], False)
expect("fsdp.sharding_strategy", fsdp["sharding_strategy"], "no_shard")
expect("fsdp.use_orig_params", fsdp["use_orig_params"], False)

# Data and evaluation protocol. total_num_envs x rollout_epoch == 512 is the
# invariant that makes the eval consume the identical fixed-reset multiset as
# every other arm (PAPER_SUMMARY 4.1); the split between the two factors only
# controls how many MuJoCo EGL contexts stack on each GPU.
expect("env.train.total_num_envs", env["train"]["total_num_envs"], 128)
expect("env.train.rollout_epoch", env["train"]["rollout_epoch"], 2)
expect("env.train.use_ordered_reset_state_ids", env["train"]["use_ordered_reset_state_ids"], True)
expect("env.eval.use_fixed_reset_state_ids", env["eval"]["use_fixed_reset_state_ids"], True)
eval_slots = env["eval"]["total_num_envs"] * env["eval"]["rollout_epoch"]
expect("env.eval reset slots", eval_slots, 512)
expect("actor.model.model_path", actor["model"]["model_path"], sft_model)
expect("rollout.model.model_path", cfg["rollout"]["model"]["model_path"], sft_model)

# Gradient accumulation has to divide exactly, or FSDP silently drops samples.
world_size = 8
micro, glob = actor["micro_batch_size"], actor["global_batch_size"]
if glob % (micro * world_size) != 0:
    failures.append(
        f"global_batch_size {glob} is not divisible by "
        f"micro_batch_size {micro} x {world_size} ranks"
    )

expert_rank = gse["total_rank"] // gse["num_experts"]
scaling = float(gse["lora_alpha"]) / gse["total_rank"]
print(
    "[preflight] no-router GSE: "
    f"total_rank={gse['total_rank']} "
    f"experts={gse['num_experts']} (all generalized) "
    f"rank/expert={expert_rank} active_rank={gse['total_rank']} "
    f"scaling={scaling:.4f} gate_mass=1.0"
)
print(
    "[preflight] budget: "
    f"{runner['max_epochs']} steps, save every {runner['save_interval']}, "
    f"eval every {runner['val_check_interval']} over {eval_slots} reset slots, "
    f"accum={glob // (micro * world_size)}, lr={optim['lr']}"
)
if scaling != 1.0:
    print(
        f"[preflight] NOTE scaling is {scaling:.4f}, not 1.0 -- this is the "
        "gate-mass-matched companion arm, not the lr-matched ablation"
    )

if failures:
    print("[preflight] resolved config does not match the ablation protocol:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    sys.exit(1)
print("[preflight] resolved config matches the ablation protocol")
PY

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "[preflight] PREFLIGHT_ONLY=1; resolved config kept at $RESOLVED, not training"
  exit 0
fi

# --- training loop -----------------------------------------------------------
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  # Crash recovery only. resume_dir keeps the SAVED optimizer and LR scheduler,
  # which is what a mid-run restart wants; reset_lr_scheduler_on_resume is
  # deliberately NOT passed, so it stays at its False default. Do not add it
  # here -- re-applying the configured lr would restart the cosine from step 0
  # and change the schedule this arm is supposed to share with the main method.
  RESUME=null
  if [ -d "$CKPT_ROOT" ]; then
    latest=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null |
      sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -n "${latest:-}" ] && [ -d "$CKPT_ROOT/global_step_$latest/actor" ]; then
      RESUME="$CKPT_ROOT/global_step_$latest"
    fi
  fi
  echo "[loop] attempt $attempt/$MAX_ATTEMPTS resume_dir=$RESUME $(date -Is)" |
    tee -a "$RUN_DIR/console.log"

  python "$EMBODIED_PATH/train_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name "$CONFIG" \
    "${BASE_OVERRIDES[@]}" \
    "runner.resume_dir=$RESUME" \
    +runner.eval_on_resume=true \
    "${WANDB_OVERRIDES[@]}" \
    2>&1 | tee -a "$RUN_DIR/console.log"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "[loop] training completed normally on attempt $attempt" |
      tee -a "$RUN_DIR/console.log"
    exit 0
  fi
  echo "[loop] training exited rc=$rc; retrying from the latest checkpoint in ${RETRY_SLEEP}s" |
    tee -a "$RUN_DIR/console.log"
  sleep "$RETRY_SLEEP"
done

echo "[loop] giving up after $MAX_ATTEMPTS attempts" | tee -a "$RUN_DIR/console.log"
exit 1
