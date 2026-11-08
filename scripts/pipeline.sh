#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/1_generate_timestamps.sh"
bash "$SCRIPT_DIR/2_generate_transcripts.sh"
