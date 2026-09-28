#!/usr/bin/env bash
# Re-score, analyse and redraw everything from the saved raw responses (no API calls).
# Uses $BLEND_DIR if set (e.g. BLEND_DIR=../BLEnD), otherwise clones BLEnD into ./external/BLEnD
# at the commit the experiment used. (Not ./BLEnD: on macOS that name collides with the code folder blend/.)
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=7b9c131719e7fe5f9bed0f8b855532d613cc9f2b
if [ -z "${BLEND_DIR:-}" ]; then
  [ -d external/BLEnD/data ] || git clone -q https://github.com/nlee0212/BLEnD.git external/BLEnD
  git -C external/BLEnD checkout -q "$COMMIT"
  export BLEND_DIR="$PWD/external/BLEnD"
fi
echo "BLEnD: $BLEND_DIR"
python -m blend.score
python -m blend.analyze
python -m blend.validation
python -m blend.response_language
python -m blend.manual_sample --errors-only
python -m blend.figures
