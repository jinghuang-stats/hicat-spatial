#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-checkpoints/uni}"

python -m pip install -U huggingface_hub

echo "If you have not logged in to Hugging Face yet, run:"
echo "  huggingface-cli login"
echo ""
echo "Downloading UNI checkpoint to: ${CHECKPOINT_DIR}"

python - "$CHECKPOINT_DIR" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

local_dir = Path(sys.argv[1]).expanduser().resolve()
local_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = hf_hub_download(
    repo_id="MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir=str(local_dir),
    force_download=False,
)

print(f"UNI checkpoint downloaded to: {checkpoint_path}")
PY