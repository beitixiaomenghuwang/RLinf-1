#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --exclude=paraai-n32-h-01-agent-92,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-87,paraai-n32-h-01-agent-153,paraai-n32-h-01-agent-74,paraai-n32-h-01-agent-75
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=slurm-theory-sft-highk-4gpu-%j.out

# High-k SFT evaluation on 4 GPUs for pass@k analysis.
#
# PROTOCOL
# --------
# 16 envs × 160 epochs = 2560 total slots. MT50 has 50 tasks × 10 reset states
# = 500 states, so 2560 slots gives each task floor(2560/50) = 51 trials.
# That supports pass@k up to k=40 with the unbiased HumanEval estimator.
#
# envs/GPU = 16/4 = 4, matching the 8-GPU training baseline (job 1475511) and
# keeping MuJoCo EGL contexts at 4 per GPU (~123 MiB each, within safe range).
#
# The reset sequence is shuffle=False, so the flat ordering is deterministic and
# process-count-independent. The 4-GPU version consumes the same 2560-slot
# multiset of reset states as the 2-GPU version (8 envs×320).
#
# NOTE: This job is for pass@k analysis, not for reproducing the main-table
# point estimate (which uses the 512-slot protocol in sft_eval.sbatch).

set -euo pipefail

echo "launcher start: node=$(hostname) job=${SLURM_JOB_ID:-none}"

NUM_NODES=1
NUM_GPUS=${NUM_GPUS:-4}
EVAL_ENVS=${EVAL_ENVS:-16}
EVAL_EPOCH=${EVAL_EPOCH:-160}
CONFIG=${CONFIG:-metaworld_50_openpi_pi05_eval}
EXP_NAME=${EXP_NAME:-metaworld50_sft_highk_eval_8gpu}

GXY=${GXY:-$HOME/guoxueyang}
REPO_PATH=${REPO_PATH:-$GXY/RLinf-1}
SIF_IMAGE=${SIF_IMAGE:-$GXY/rlinf-arm64-ngc2412-verified.sif}
RLINF_VENV=${RLINF_VENV:-$GXY/rlinf-openpi-metaworld-arm64-venv}
SFT_MODEL=${SFT_MODEL:-$REPO_PATH/model/RLinf-Pi05-MetaWorld-SFT}
SFT_SHA256=${SFT_SHA256:-6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_PATH/parallel_output}
RUN_DIR=$OUTPUT_ROOT/${EXP_NAME}_${SLURM_JOB_ID}
RAY_DIR=$RUN_DIR/ray

test -d "$REPO_PATH" || { echo "Missing REPO_PATH=$REPO_PATH" >&2; exit 3; }
test -f "$SIF_IMAGE" || { echo "Missing SIF_IMAGE=$SIF_IMAGE" >&2; exit 3; }
test -d "$RLINF_VENV" || { echo "Missing RLINF_VENV=$RLINF_VENV" >&2; exit 3; }
actual=$(sha256sum "$SFT_MODEL/model.safetensors" | awk '{print $1}')
test "$actual" = "$SFT_SHA256" || { echo "Unexpected SFT SHA256: $actual" >&2; exit 3; }

python3 - "$EVAL_ENVS" "$EVAL_EPOCH" "$NUM_GPUS" <<'PY'
import sys
envs, epoch, procs = (int(x) for x in sys.argv[1:4])
if envs % procs:
    raise SystemExit(f"total_num_envs {envs} is not divisible by {procs} ranks")
n_trials = envs * epoch // 50
if envs * epoch < 2500:
    raise SystemExit(f"total slots {envs * epoch} gives only {n_trials} trials/task; need >=50")
if envs // procs > 4:
    raise SystemExit(
        f"{envs // procs} envs per rank exceeds 4 EGL contexts/GPU; would OOM")
print(f"high-k protocol OK: {procs} GPUs x {envs // procs} envs x {epoch} epochs "
      f"= {envs * epoch} total slots, ~{n_trials} trials/task")
PY

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
test "${drv%%.*}" -ge 525 || { echo "$(hostname) has driver $drv, need >= 525" >&2; exit 5; }
echo "node=$(hostname) driver=$drv gpus=$(nvidia-smi -L | wc -l)"

mkdir -p "$RUN_DIR" "$RAY_DIR"

module purge
module load singularity/3.9.2-with-suid

bind_args=(
  --bind "$REPO_PATH:/workspace/RLinf"
  --bind "$RLINF_VENV:/workspace/venv"
  --bind "$RUN_DIR:/workspace/run"
  --bind "$RAY_DIR:/workspace/ray"
)
for egl_dir in /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do
  test -d "$egl_dir" && bind_args+=(--bind "$egl_dir:$egl_dir")
done
for egl_lib in /usr/lib64/libEGL.so.1 /usr/lib64/libGLX.so.0; do
  test -e "$egl_lib" && bind_args+=(--bind "$egl_lib:$egl_lib")
done

export SINGULARITYENV_MUJOCO_GL=egl
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
export SINGULARITYENV_RAY_memory_monitor_refresh_ms=0
export SINGULARITYENV_MALLOC_ARENA_MAX=4
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

singularity exec --nv "${bind_args[@]}" "$SIF_IMAGE" bash -lc '
  set -euo pipefail
  source /workspace/venv/bin/activate
  export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
  export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
  export HYDRA_FULL_ERROR=1
  python -u /workspace/RLinf/evaluations/eval_embodied_agent.py \
    --config-path /workspace/RLinf/evaluations/metaworld/ \
    --config-name "'"$CONFIG"'" \
    cluster.num_nodes='"$NUM_NODES"' \
    runner.logger.experiment_name="'"$EXP_NAME"'" \
    runner.logger.log_path=/workspace/run \
    env.eval.total_num_envs='"$EVAL_ENVS"' \
    env.eval.rollout_epoch='"$EVAL_EPOCH"' \
    rollout.model.model_path=/workspace/RLinf/model/RLinf-Pi05-MetaWorld-SFT
'

echo "done: $RUN_DIR"
