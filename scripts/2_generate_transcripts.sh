#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_DIR=${TRANSCRIPTS_DIR:-./results/transcripts_with_lang_whisper}
mkdir -p "$OUTPUT_DIR"
RANK_DIR=$(mktemp -d "$OUTPUT_DIR/.ranks.XXXXXX")
trap 'rm -rf -- "$RANK_DIR"' EXIT
torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" bin/generate_transcripts.py \
    --annotation-dir "${TIMESTAMPS_DIR:-./results/timestamps}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --target-sr 16000 \
    --asr-model-dir "${ASR_MODEL_DIR:-openai/whisper-large-v3}" \
    --output-dir "$RANK_DIR" "$@"
python bin/merge.py --data-dir "$RANK_DIR" --output-dir "$OUTPUT_DIR"
