#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Run when shell networking is available. Installs only into this workspace.
task_uv_install_dir="$PWD/refresh_20261103/runtime/bin"
mkdir -p "$task_uv_install_dir"
curl --fail --location --show-error https://astral.sh/uv/install.sh -o refresh_20261103/runtime/install_uv.sh
UV_INSTALL_DIR="$task_uv_install_dir" UV_NO_MODIFY_PATH=1 sh refresh_20261103/runtime/install_uv.sh
export UV_PYTHON_INSTALL_DIR="$PWD/refresh_20261103/runtime/python"
export UV_CACHE_DIR="$PWD/refresh_20261103/runtime/cache"
"$task_uv_install_dir/uv" python install 3.13
"$task_uv_install_dir/uv" venv --python 3.13 .venv
"$task_uv_install_dir/uv" pip install --python .venv/bin/python -r refresh_20261103/requirements.txt
.venv/bin/python -c 'import numpy,pandas,scipy,sklearn,joblib,matplotlib; print("Scientific Python imports passed")'
