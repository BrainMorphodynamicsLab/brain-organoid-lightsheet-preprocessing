#!/bin/zsh
set -u
cd "$(dirname "$0")"

CANDIDATES=(
  "${LIGHTSHEET_PYTHON:-}"
  "$PWD/.venv/bin/python"
  "/opt/anaconda3/envs/lightsheet/bin/python"
  "$HOME/anaconda3/envs/lightsheet/bin/python"
  "$HOME/miniconda3/envs/lightsheet/bin/python"
  "$HOME/mambaforge/envs/lightsheet/bin/python"
  "$HOME/miniforge3/envs/lightsheet/bin/python"
)

PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "A compatible lightsheet Python environment was not found."
  echo "Create it from environment.yml, then try again."
  echo
  echo "Expected one of:"
  printf '  %s\n' "${CANDIDATES[@]}"
  echo
  read -k 1 "?Press any key to close..."
  exit 1
fi

echo "Using Python: $PYTHON"
"$PYTHON" -c 'import numpy, scipy, tifffile, imagecodecs, skimage, cv2, matplotlib, PIL' || {
  echo "The environment exists but required imports failed."
  read -k 1 "?Press any key to close..."
  exit 1
}

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "$PYTHON" gui/script_gui.py full_pipeline
else
  exec "$PYTHON" gui/script_gui.py full_pipeline
fi
