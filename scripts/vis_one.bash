#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT_DIR}"

python qwen_gem/cli.py \
  --model_path "lmms-lab/LLaVA-Video-7B-Qwen2" \
  --results_dir "qwen_gem/outputs/debug_one" \
  --one_video "${1:?need_video_path}" \
  --one_question "${2:-A photo of a person whisk eggs.}" \
  --max_frames 16
