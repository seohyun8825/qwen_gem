#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"

MODEL_PATH="${1:-lmms-lab/LLaVA-Video-7B-Qwen2}"
RESULTS_DIR="${2:-qwen_gem/outputs/videomme_infer}"

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python -m llava.eval.infer \
  --model_path "${MODEL_PATH}" \
  --results_dir "${RESULTS_DIR}" \
  --data_path "/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/formatted_dataset_10.json" \
  --video_root "/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/videos/data" \
  --dataset_name "VideoMME" \
  --max_frames_num 48 \
  --test_ratio 1 \
  --cals_acc \
  --no_cache

RESULTS_JSON="${RESULTS_DIR}/results.json"
if [[ -f "${RESULTS_JSON}" ]]; then
  python - "$RESULTS_JSON" <<'PY'
import json
import sys

results_path = sys.argv[1]
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

overall = data.get("all", {})
total = overall.get("total", 0)
correct = overall.get("correct", 0)
accuracy = overall.get("accuracy")

if accuracy is None and total:
    accuracy = correct / max(total, 1)

if accuracy is not None:
    print(f"[Summary] Total: {total} | Correct: {correct} | Accuracy: {accuracy:.4f}")
else:
    print(f"[Summary] Total: {total} | Correct: {correct}")
PY
else
  echo "[Summary] results.json not found at ${RESULTS_JSON}"
fi
