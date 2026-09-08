#!/bin/bash
# Submit the MetaWorld MT50 rank-64 NO-ROUTER multi-expert arm on this server's
# 8 x A100-SXM4-80GB (PAPER_SUMMARY.md 4.4 ablation A, the P0 unrun experiment
# that turns section 5.3's routing dependence LOWER BOUNDS into a contribution
# number).
#
#   config     examples/embodiment/config/
#                metaworld_50_ppo_openpi_pi05_gse_action_r32_svd_no_router.yaml
#   arm        8 experts / rank 64 (rank 8 each) / uniform aggregation / no
#              router parameters / orthogonal_zero / scaling 1.0 / lr 5e-5
#   budget     300 steps, save every 10 (adapter only, no retention cap),
#              evaluate every 20 over 512 fixed reset slots
#   capacity   128 train envs x 2 rollout epochs, actor micro 128 / global 1024,
#              rollout micro 8 -- the values measured on THIS machine for the
#              rank-64 2g6s run gse-formal-seed42 (HANDOFF.md 9.1/9.2)
#
# Run it from the host, ideally inside tmux so an ssh drop does not kill it:
#   tmux new -s mw-no-router
#   bash /home/xueyang/RLinf/scripts/launch_metaworld_gse_r64_no_router_8gpu.sh
set -euo pipefail

# Host paths on the 8-GPU server (Linux). These were /Users/xueyang/... until
# 2026-09-08; that is a macOS layout and does not exist here, so every launch
# aborted at the `test -d "$repo_host"` preflight below.
repo_host=${REPO_HOST:-/home/xueyang/RLinf}
sft_host=${SFT_HOST:-/DATA/disk0/xueyang/model/RLinf-Pi05-MetaWorld-SFT}
# Outputs go to disk1, not the disk0 pi05-gse root that holds the older runs.
output_host=${OUTPUT_HOST:-/DATA/disk1/xueyang/model/pi05-gse}
ray_host=${RAY_HOST:-/DATA/disk1/xueyang/Data/rlinf-ray-metaworld-no-router}
cache_host=${CACHE_HOST:-/home/xueyang/RLinf/cache/huggingface}
netrc_host=${NETRC_HOST:-/home/xueyang/.netrc}
image=${IMAGE:-rlinf/rlinf:agentic-rlinf0.3-maniskill_libero}

experiment=${EXP_NAME:-metaworld50_gse_r64_no_router_300_8gpu_seed42}
container=${CONTAINER_NAME:-rlinf-pi05-gse-no-router}
min_free_gib=${MIN_FREE_GIB:-60}

test -d "$repo_host" || { echo "[preflight] missing repo $repo_host" >&2; exit 3; }
test -f "$sft_host/model.safetensors" || {
  echo "[preflight] missing SFT weights $sft_host/model.safetensors" >&2
  exit 3
}
test -d "$(dirname "$output_host")" || {
  echo "[preflight] missing output parent $(dirname "$output_host")" >&2
  exit 3
}

# The 300-step run writes 30 adapter-only checkpoints plus tensorboard and Ray
# spill. Adapter-only checkpoints are ~110 MiB at rank 32, so this is tens of
# GiB, not the ~500 GiB the old full-weight checkpoints would have needed.
mkdir -p "$output_host" "$ray_host/session" "$ray_host/spill" "$ray_host/tmp"
free_gib=$(df -BG --output=avail "$output_host" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$free_gib" ]; then
  free_gib=$(df -BG "$output_host" | tail -1 | awk '{gsub("G","",$4); print $4}')
fi
if [ -z "$free_gib" ] || [ "$free_gib" -lt "$min_free_gib" ]; then
  echo "[preflight] only ${free_gib:-?}GiB free under $output_host, need >= ${min_free_gib}GiB" >&2
  exit 3
fi

bash "$repo_host/scripts/check_nvidia_egl_health.sh" 8

# Refuse to pile onto someone else's job. This is not hypothetical: on
# 2026-09-08 all eight GPUs were at 100% utilisation under another user's conda
# python processes (~9.7 GiB each) while `docker ps` showed only dcgm-exporter,
# so a naive launch would have OOM-raced them. sharding_strategy is "no_shard",
# so every rank needs a full model replica and there is no room to share a card.
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader 2>/dev/null || true)
if [ -n "$busy" ]; then
  echo "[preflight] GPUs already have compute processes:" >&2
  printf '  %s\n' "$busy" >&2
  if [ "${ALLOW_BUSY_GPUS:-0}" != "1" ]; then
    echo "[preflight] refusing to launch. Wait for them, or set ALLOW_BUSY_GPUS=1 if you own them." >&2
    exit 3
  fi
  echo "[preflight] ALLOW_BUSY_GPUS=1 set; continuing anyway" >&2
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
  echo "[preflight] a container named $container already exists; remove it or set CONTAINER_NAME" >&2
  exit 3
fi

tty_flags=(-i)
if [ -t 0 ]; then
  tty_flags=(-it)
fi

echo "[launch] experiment=$experiment"
echo "[launch] run dir (host)=$output_host/$experiment"

exec docker run "${tty_flags[@]}" --rm --privileged \
  --gpus all \
  --shm-size 256g \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "$container" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e RLINF_RAY_TEMP_DIR=/workspace/ray/session \
  -e RLINF_RAY_OBJECT_SPILL_DIR=/workspace/ray/spill \
  -e TMPDIR=/workspace/ray/tmp \
  -e WANDB_ENTITY="${WANDB_ENTITY:-gxy1000h-jilin-university}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-pi05-multitask-peft-rl}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  -e EXP_NAME="$experiment" \
  -e RUN_DIR="/workspace/output/$experiment" \
  -e SFT_MODEL=/workspace/models/RLinf-Pi05-MetaWorld-SFT \
  -e ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-128}" \
  -e ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-1024}" \
  -e ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-8}" \
  -e SEED="${SEED:-42}" \
  -e GSE_LORA_ALPHA="${GSE_LORA_ALPHA:-}" \
  -e ACTOR_LR="${ACTOR_LR:-}" \
  -e MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}" \
  -v "$repo_host":/workspace/RLinf \
  -v "$sft_host":/workspace/models/RLinf-Pi05-MetaWorld-SFT:ro \
  -v "$output_host":/workspace/output \
  -v "$ray_host":/workspace/ray \
  -v "$cache_host":/root/.cache/huggingface \
  -v "$netrc_host":/root/.netrc:ro \
  -w /workspace/RLinf \
  "$image" \
  bash -lc '
    set -euo pipefail
    source switch_env openpi
    export REPO_PATH=/workspace/RLinf
    export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
    export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    mkdir -p "$RUN_DIR"
    # One EGL device per CUDA ordinal, all eight distinct. Each MuJoCo context
    # costs ~123 MiB and train+eval contexts coexist, so a collapsed mapping
    # shows up later as "Offscreen framebuffer is not complete, error 0x8cdd"
    # rather than as an error here.
    python -c '\''from rlinf.scheduler.hardware.accelerators.nvidia_gpu import _query_egl_index_by_cuda_ordinal as q; m = q(); assert len(m) == 8, m; assert len(set(m.values())) == 8, m; print("[egl-preflight] CUDA ordinal -> EGL index:", m)'\''
    exec bash /workspace/RLinf/scripts/run_metaworld_gse_r64_no_router_8gpu_loop.sh
  '
