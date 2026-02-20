#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${HOME}/.local/bin/speak"

mkdir -p "${HOME}/.local/bin"
ln -sf "${SCRIPT_DIR}/bin/speak" "${TARGET}"
echo "Installed: ${TARGET} -> ${SCRIPT_DIR}/bin/speak"
