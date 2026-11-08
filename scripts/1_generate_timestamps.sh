#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_DIR=${TIMESTAMPS_DIR:-./results/timestamps}
mkdir -p "$OUTPUT_DIR"
RANK_DIR=$(mktemp -d "$OUTPUT_DIR/.ranks.XXXXXX")
trap 'rm -rf -- "$RANK_DIR"' EXIT
torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" bin/generate_timestamps.py \
    --audio_dir "${AUDIO_DIR:-./sources}" \
    --output_dir "$RANK_DIR" \
    --batch_size "${BATCH_SIZE:-4}" \
    --sampling_rate 16000 \
    --threshold 0.5 \
    --min_silence_duration_ms 250 \
    --min_speech_duration_ms 500 "$@"
python bin/merge.py --data-dir "$RANK_DIR" --output-dir "$OUTPUT_DIR"
