#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT_DIR}"

python qwen_gem/cli.py \
  --model_path "${1:-lmms-lab/LLaVA-Video-7B-Qwen2}" \
  --results_dir "qwen_gem/outputs/VideoMME" \
  --data_path "DATAS/eval/VideoMME/formatted_dataset_10.json" \
  --video_root "DATAS/eval/VideoMME/videos/data" \
  --max_frames 16 \
  --topk_frames 8
