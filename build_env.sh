#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${BRAINWM_ENV_NAME:-brainwm}"
PYTHON_VERSION="${BRAINWM_PYTHON_VERSION:-3.10.18}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl#sha256=c59be18fa934e132e5a405bcca6673af1d9d09a0036a9e081133bc7b7fc2992e"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

RUN=(conda run --no-capture-output -n "${ENV_NAME}")

"${RUN[@]}" python -m pip install --upgrade \
    "pip==25.2" "setuptools==75.8.0" wheel

"${RUN[@]}" python -m pip install \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url "${PYTORCH_INDEX}"

"${RUN[@]}" python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
"${RUN[@]}" python -m pip install "${FLASH_WHEEL}"

"${RUN[@]}" python - <<'PY'
import importlib
import importlib.metadata as metadata
import torch

expected = {
    "accelerate": "1.11.0",
    "deepspeed": "0.18.0",
    "diffusers": "0.31.0",
    "transformers": "4.47.0",
    "huggingface-hub": "0.36.0",
    "peft": "0.17.1",
    "monai": "1.5.1",
    "linear-attention-transformer": "0.19.1",
    "numpy": "1.26.4",
    "pandas": "1.5.3",
}
errors = []
for package, wanted in expected.items():
    got = metadata.version(package)
    if got != wanted:
        errors.append(f"{package}: expected {wanted}, got {got}")

if torch.__version__ != "2.5.1+cu121":
    errors.append(f"torch: expected 2.5.1+cu121, got {torch.__version__}")
if torch.version.cuda != "12.1":
    errors.append(f"torch CUDA runtime: expected 12.1, got {torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("torch.cuda.is_available() is False")

for module in (
    "accelerate", "deepspeed", "diffusers", "transformers", "peft",
    "flash_attn", "monai", "linear_attention_transformer", "decord",
):
    importlib.import_module(module)

if errors:
    raise SystemExit("Environment validation failed:\n- " + "\n- ".join(errors))
print(f"Environment OK: torch={torch.__version__}, CUDA={torch.version.cuda}, GPUs={torch.cuda.device_count()}")
PY

echo "BrainWM environment '${ENV_NAME}' is ready."
