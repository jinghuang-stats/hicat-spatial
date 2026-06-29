mkdir -p hicat/preprocessing/uni/checkpoints

pip install -U huggingface_hub

huggingface-cli login

python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

local_dir = Path("hicat/preprocessing/uni/checkpoints")
local_dir.mkdir(parents=True, exist_ok=True)

hf_hub_download(
    repo_id="MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir=str(local_dir),
    force_download=False,
)

print(f"UNI checkpoint downloaded to: {local_dir / 'pytorch_model.bin'}")
PY