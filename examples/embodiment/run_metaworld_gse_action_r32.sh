#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_NAME=metaworld_50_ppo_openpi_pi05_gse_action_r32
SFT_MODEL="${SFT_MODEL:-/workspace/models/RLinf-Pi05-MetaWorld-SFT}"
RUN_DIR="${RUN_DIR:-/workspace/output/gse-action-r32-seed42}"
EXP_NAME="${EXP_NAME:-gse_action_r32_seed42}"
SEED="${SEED:-42}"

export EMBODIED_PATH="${EMBODIED_PATH:-$SCRIPT_DIR}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR"

python "$SCRIPT_DIR/train_embodied_agent.py" \
  --config-path "$SCRIPT_DIR/config" \
  --config-name "$CONFIG_NAME" \
  "$@" \
  actor.micro_batch_size=16 \
  actor.global_batch_size=1024 \
  actor.model.model_path="$SFT_MODEL" \
  rollout.model.model_path="$SFT_MODEL" \
  runner.resume_dir=null \
  runner.logger.log_path="$RUN_DIR" \
  runner.logger.experiment_name="$EXP_NAME" \
  actor.seed="$SEED" \
  rollout.seed="$SEED" \
  2>&1 | tee -a "$RUN_DIR/console.log"
