#!/bin/bash
# Fresh Linux box -> the established GPT-OSS/MXFP4 dependency window.
# Native quantization still needs mxfp4_guard's actual loader/layout checks.
# Default keeps the existing python3/active-venv entry point used by bootstrap.
set -euo pipefail

python_bin=${SHARD_SETUP_PYTHON:-python3}
venv_dir=""
while (($#)); do
    case "$1" in
        --venv)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "setup_box: --venv requires a directory" >&2
                exit 2
            fi
            venv_dir=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash phase0/setup_box.sh [--venv DIRECTORY]"
            echo "Default: install into python3 (or SHARD_SETUP_PYTHON), preserving an active venv."
            echo "--venv: create/reuse an isolated venv; launch the engine with its bin/python."
            exit 0
            ;;
        *)
            echo "setup_box: unknown option" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$venv_dir" ]]; then
    if [[ -e "$venv_dir" && ! -f "$venv_dir/pyvenv.cfg" ]]; then
        echo "setup_box: refusing to reuse a non-venv directory" >&2
        exit 2
    fi
    if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
        "$python_bin" -m venv "$venv_dir"
    fi
    python_bin="$venv_dir/bin/python"
fi

environment_mode=$("$python_bin" - <<'PY'
import sys
print("venv" if sys.prefix != sys.base_prefix else "system")
PY
)
if [[ -n "$venv_dir" && "$environment_mode" != venv ]]; then
    echo "setup_box: requested venv interpreter is not isolated" >&2
    exit 1
fi
pip_install=(install -q)
case "$environment_mode" in
    system) pip_install+=(--break-system-packages) ;;
    venv) ;;
    *) echo "setup_box: cannot determine Python environment" >&2; exit 1 ;;
esac

if [[ "$environment_mode" == system ]]; then
    rich_without_record=$("$python_bin" - <<'PY'
from importlib import metadata
try:
    rich = metadata.distribution("rich")
except metadata.PackageNotFoundError:
    print("no")
else:
    print("yes" if rich.read_text("RECORD") is None else "no")
PY
)
    case "$rich_without_record" in
        yes)
            # Debian owns its rich files without pip's uninstall RECORD. Install
            # a pip-managed overlay without removing those files. Keep the bypass
            # scoped to rich; the normal dependency resolver handles its deps.
            "$python_bin" -m pip "${pip_install[@]}" --ignore-installed --no-deps rich
            # A custom pip target/user path or PYTHONPATH can leave Debian's
            # metadata ahead of the new overlay. Confirm the same interpreter
            # selects a RECORD-backed distribution before any other installs.
            "$python_bin" - <<'PY'
from importlib import metadata
import sys
try:
    record = metadata.distribution("rich").read_text("RECORD")
except (metadata.PackageNotFoundError, OSError, ValueError):
    record = None
if not record:
    print("setup_box: pip rich overlay is not selected by this Python; use --venv DIRECTORY and its bin/python, or correct the pip target/Python path", file=sys.stderr)
    sys.exit(1)
PY
            ;;
        no) ;;
        *) echo "setup_box: cannot inspect rich installation metadata" >&2; exit 1 ;;
    esac
fi

"$python_bin" -m pip "${pip_install[@]}" --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
"$python_bin" -m pip "${pip_install[@]}" "transformers>=5.19.0,<5.20.0" huggingface_hub safetensors accelerate "kernels>=0.17.0,<0.18.0" triton
"$python_bin" - <<'PY'
import torch, transformers, kernels
print("torch", torch.__version__, "| tf", transformers.__version__,
      "| kernels", kernels.__version__, "| gpus", torch.cuda.device_count())
PY
if [[ -n "$venv_dir" ]]; then
    printf 'setup_box: isolated runtime Python: %s\n' "$python_bin"
fi
