#!/usr/bin/env bash
# Test everything with one command (see tools/test_all.py for the list of checks).
#   ./test_all.sh            # no API keys needed
#   ./test_all.sh --api      # + 1 real API call per model (needs OPENAI_API_KEY / GEMINI_API_KEY)
# Uses $BLEND_DIR if set, otherwise ./BLEnD or ../BLEnD.
cd "$(dirname "$0")"
exec python tools/test_all.py "$@"
