#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="lmms-lab/LLaVA-Video-7B-Qwen2"
DATA_PATH="/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/formatted_dataset_10.json"
VIDEO_ROOT="/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/videos/data"
OUT_ROOT="/home/video_understanding/qwen_gem/outputs/retrieval_qa"

SHOT_BASED="true"
NUM_QUERIES=3
QUERY_STAGE_FRAMES=24
CLIP_FPS=1
PER_QUERY_TOPK=1
MIN_GAP_SEC=0.5

FINAL_MAX_FRAMES=24
MULTIPROCESS="false"
NO_CACHE="true"

print_usage() {
  cat <<USAGE
Usage: $0 [--model PATH] [--data JSON] [--videos DIR] [--out DIR]
          [--shot_based true|false] [--num_queries N] [--query_frames N]
          [--clip_fps F] [--per_query_topk K] [--min_gap_sec S]
          [--final_max_frames N] [--multiprocess true|false] [--no_cache true|false]

Examples:
  $0 --shot_based true  --num_queries 3
  $0 --shot_based false --clip_fps 2 --num_queries 5
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2;;
    --data) DATA_PATH="$2"; shift 2;;
    --videos) VIDEO_ROOT="$2"; shift 2;;
    --out) OUT_ROOT="$2"; shift 2;;
    --shot_based) SHOT_BASED="$2"; shift 2;;
    --num_queries) NUM_QUERIES="$2"; shift 2;;
    --query_frames) QUERY_STAGE_FRAMES="$2"; shift 2;;
    --clip_fps) CLIP_FPS="$2"; shift 2;;
    --per_query_topk) PER_QUERY_TOPK="$2"; shift 2;;
    --min_gap_sec) MIN_GAP_SEC="$2"; shift 2;;
    --final_max_frames) FINAL_MAX_FRAMES="$2"; shift 2;;
    --multiprocess) MULTIPROCESS="$2"; shift 2;;
    --no_cache) NO_CACHE="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "[ERR] Unknown arg: $1"; print_usage; exit 1;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

QUERIES_DIR="${OUT_ROOT}/queries"
SELECTED_DIR="${OUT_ROOT}/selected_frames"
FINAL_DIR="${OUT_ROOT}/infer_final"

mkdir -p "${QUERIES_DIR}" "${SELECTED_DIR}" "${FINAL_DIR}"

echo "[CFG] MODEL_PATH=${MODEL_PATH}"
echo "[CFG] DATA_PATH=${DATA_PATH}"
echo "[CFG] VIDEO_ROOT=${VIDEO_ROOT}"
echo "[CFG] OUT_ROOT=${OUT_ROOT}"
echo "[CFG] SHOT_BASED=${SHOT_BASED}, NUM_QUERIES=${NUM_QUERIES}, QUERY_STAGE_FRAMES=${QUERY_STAGE_FRAMES}"
echo "[CFG] CLIP_FPS=${CLIP_FPS}, PER_QUERY_TOPK=${PER_QUERY_TOPK}, MIN_GAP_SEC=${MIN_GAP_SEC}"
echo "[CFG] FINAL_MAX_FRAMES=${FINAL_MAX_FRAMES}, MULTIPROCESS=${MULTIPROCESS}, NO_CACHE=${NO_CACHE}"

echo ""
echo "==> [Stage 1] Generating ranked, visually explicit scene queries..."
GEN_CMD=(
  python -m llava.eval.gen_queries
  --model_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --video_root "${VIDEO_ROOT}"
  --out_dir "${QUERIES_DIR}"
  --num_queries "${NUM_QUERIES}"
  --query_stage_frames "${QUERY_STAGE_FRAMES}"
)
if [[ "${SHOT_BASED}" == "true" ]]; then
  GEN_CMD+=(--shot_based)
fi
if [[ "${NO_CACHE}" == "true" ]]; then
  GEN_CMD+=(--no_cache)
fi
"${GEN_CMD[@]}"

echo ""
echo "==> [Stage 2] Ranking/selecting frames per query with CLIP..."
RANK_CMD=(
  python -m llava.eval.rank_frames
  --data_path "${DATA_PATH}"
  --video_root "${VIDEO_ROOT}"
  --queries_dir "${QUERIES_DIR}"
  --selected_dir "${SELECTED_DIR}"
  --per_query_topk "${PER_QUERY_TOPK}"
  --min_gap_sec "${MIN_GAP_SEC}"
  --uniform_fps "${CLIP_FPS}"
)
if [[ "${SHOT_BASED}" == "true" ]]; then
  RANK_CMD+=(--shot_based)
fi
if [[ "${NO_CACHE}" == "true" ]]; then
  RANK_CMD+=(--no_cache)
fi
"${RANK_CMD[@]}"

MP_FLAG=""
if [[ "${MULTIPROCESS}" == "true" ]]; then
  MP_FLAG="--multiprocess"
fi

echo ""
echo "==> [Stage 3] Final high-resolution inference & evaluation..."
INFER_CMD=(
  python -m llava.eval.infer
  --model_path "${MODEL_PATH}"
  --results_dir "${FINAL_DIR}"
  --data_path "${DATA_PATH}"
  --video_root "${VIDEO_ROOT}"
  --dataset_name "VideoMME"
  --max_frames_num "${FINAL_MAX_FRAMES}"
  --selected_frames_root "${SELECTED_DIR}"
  --cals_acc
)
if [[ "${MP_FLAG}" == "--multiprocess" ]]; then
  INFER_CMD+=(--multiprocess)
fi
if [[ "${NO_CACHE}" == "true" ]]; then
  INFER_CMD+=(--no_cache)
fi
"${INFER_CMD[@]}"

RESULTS_JSON="${FINAL_DIR}/results.json"
if [[ -f "${RESULTS_JSON}" ]]; then
  python - "$RESULTS_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("all", {})
tot = summary.get("total", 0)
cor = summary.get("correct", 0)
acc = summary.get("accuracy")
if acc is None and tot:
    acc = cor / max(1, tot)
if acc is None:
    print(f"[Summary] Total: {tot} | Correct: {cor}")
else:
    print(f"[Summary] Total: {tot} | Correct: {cor} | Accuracy: {acc:.4f}")
PY
else
  echo "[Summary] results.json not found at ${RESULTS_JSON}"
fi
