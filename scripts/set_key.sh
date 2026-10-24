#!/bin/sh
# Prompt for an API key without echoing it and store it in .env (git-ignored).
# Usage: sh scripts/set_key.sh GEMINI_API_KEY
set -eu
NAME="${1:-GEMINI_API_KEY}"
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
printf 'Paste your %s and press Enter (input is hidden): ' "$NAME"
stty -echo
IFS= read -r KEY
stty echo
printf '\n'
[ -n "$KEY" ] || { echo "No key entered; .env unchanged."; exit 1; }
KEY="$KEY" NAME="$NAME" python3 - <<'EOF'
import os, re
name, key = os.environ["NAME"], os.environ["KEY"].strip()
text = open(".env").read()
line = f"{name}={key}"
text = re.sub(rf"^{name}=.*$", lambda _: line, text, flags=re.M) if re.search(rf"^{name}=", text, re.M) else text + f"\n{line}\n"
open(".env", "w").write(text)
EOF
echo "Saved $NAME to .env (${#KEY} characters)."
