#!/usr/bin/env bash
# Run before every `git push`: looks for secrets and shows exactly what git would upload.
#   ./tools/check_before_push.sh
cd "$(dirname "$0")/.."
status=0

echo "== 1. API keys / tokens in files git would track"
files=$(git ls-files 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null)
[ -z "$files" ] && files=$(find . -type f -not -path "./.git/*" -not -path "./BLEnD/*" -not -path "./.venv/*" -not -path "./.idea/*")
hits=$(echo "$files" | tr '\n' '\0' | xargs -0 grep -I -n -E \
  "sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_-]{30,}|AQ\.[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|(OPENAI|GEMINI|GOOGLE|ANTHROPIC)_API_KEY *= *['\"]?[A-Za-z0-9_-]{16,}" 2>/dev/null)
if [ -n "$hits" ]; then echo "$hits"; echo "!! possible secret found - do NOT push"; status=1; else echo "   none"; fi

echo "== 2. files that must never be tracked"
bad=$(echo "$files" | grep -E "(^|/)\.env$|\.pem$|\.key$|id_rsa|credentials|__MACOSX|/\._|\.DS_Store$")
if [ -n "$bad" ]; then echo "$bad"; echo "!! remove these (or add them to .gitignore)"; status=1; else echo "   none"; fi

echo "== 3. BLEnD itself is not included (run_all.sh downloads it into external/)"
if echo "$files" | grep -q -E "^(\./)?external/"; then echo "!! external/ (BLEnD dataset) would be uploaded"; status=1; else echo "   ok"; fi

echo "== 4. all project files are included (catches .gitignore mistakes, e.g. case-insensitive matches on macOS)"
missing=""
for f in $(find blend tests tools -name "*.py" -o -name "*.sh" | grep -v __pycache__) README.md DATA_LICENSE.md requirements.txt run_all.sh setup.sh; do
  echo "$files" | sed 's#^\./##' | grep -qx "$f" || missing="$missing $f"
done
for d in results/Azerbaijan/raw results/Iran/raw results/manual; do
  echo "$files" | sed 's#^\./##' | grep -q "^$d/" || missing="$missing $d/"
done
if [ -n "$missing" ]; then echo "!! not included:$missing"; echo "   check with: git check-ignore -v <file>"; status=1; else echo "   ok"; fi

echo "== 5. what would be uploaded"
echo "$files" | sed 's#^\./##' | awk -F/ '{print ($2=="" ? $1 : $1"/"$2)}' | sort | uniq -c | sort -k2
echo "   total: $(echo "$files" | wc -l | tr -d ' ') files"

[ $status -eq 0 ] && echo "OK to push." || echo "Fix the problems above before pushing."
exit $status
