#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-lmms-lab/LLaVA-Video-7B-Qwen2}"
DATA_PATH="${2:-/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/formatted_dataset_10.json}"
VIDEO_ROOT="${3:-/home/video_understanding/qwen_gem/DATAS/eval/VideoMME/videos/data}"
RESULTS_DIR="${4:-/home/video_understanding/qwen_gem/outputs/VideoMME}"

cd /home/video_understanding/qwen_gem
export PYTHONPATH="/home/video_understanding/qwen_gem:${PYTHONPATH:-}"

echo "[1/2] Shot-based captioning (1 frame per shot)…"
python -m llava.eval.infer \
  --model_path "${MODEL_PATH}" \
  --results_dir "${RESULTS_DIR}" \
  --data_path "${DATA_PATH}" \
  --video_root "${VIDEO_ROOT}" \
  --dataset_name "VideoMME" \
  --shot_based \
  --qaig \
  --shot_threshold 27.0 \
  --shot_min_len 15 \
  --max_new_tokens 256 \
  --qaig_frame_threshold 0.25 \
  --qaig_conf_threshold 0.55 \
  --qaig_decay_tau 45.0 \
  --no_cache

echo "[2/2] Build & consolidate scene graphs…"
python - <<'PY'
import sys
import subprocess
import importlib
from importlib import metadata
from packaging import version

DEPS = {
    "spacy": "spacy",
    "cv2": "opencv-python",
    "networkx": "networkx",
    "sentence_transformers": "sentence-transformers",
    "rapidfuzz": "rapidfuzz",
    "scenedetect": "scenedetect",
    "open_clip": "open_clip_torch",
}

def install(pkg_spec):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg_spec],
        check=True,
    )


def ensure_exact(package_name, version_str):
    try:
        current = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        install(f"{package_name}=={version_str}")
        return
    if version.parse(current) != version.parse(version_str):
        install(f"{package_name}=={version_str}")


PINNED = {
    "transformers": "4.37.2",
    "sentence-transformers": "2.6.1",
    "huggingface-hub": "0.36.0",
}

for pkg, ver in PINNED.items():
    ensure_exact(pkg, ver)

for module_name, package_name in DEPS.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        install(package_name)
        # After installation, clear cache and import to verify availability.
        importlib.invalidate_caches()
        importlib.import_module(module_name)

import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )
PY

python /home/video_understanding/qwen_gem/scripts/graph_build.py \
  --inputs "${RESULTS_DIR}/outputs" \
  --outdir "${RESULTS_DIR}/graphs" \
  --sim_threshold 0.78

echo
echo "Done ✅"
echo "• Shot JSON : ${RESULTS_DIR}/outputs/*_shot*.json"
echo "• Graphs    : ${RESULTS_DIR}/graphs/*.json"
