#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/training/artifacts"
DEST="$ROOT/artifacts"

if [[ ! -d "$SOURCE" ]]; then
  printf 'Windows trainer artifact folder not found: %s\n' "$SOURCE" >&2
  exit 1
fi

mkdir -p "$DEST"
shopt -s nullglob
files=("$SOURCE"/*.pkl "$SOURCE"/*.gguf)
if (( ${#files[@]} == 0 )); then
  printf 'No .pkl or .gguf files found in %s\n' "$SOURCE" >&2
  exit 1
fi
# Invalidate approval before copying, including if an import is interrupted.
python3 - "$ROOT" <<'PYTHON'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from backend.release import invalidate_readiness
invalidate_readiness(root / "artifacts", "Windows artifacts imported; rerun both audits")
PYTHON
for source_file in "${files[@]}"; do
  [[ -f "$source_file" ]] || continue
  cp -f -- "$source_file" "$DEST/$(basename "$source_file")"
  printf 'Copied %s\n' "$(basename "$source_file")"
done

gguf_files=("$DEST"/*.gguf)
if (( ${#gguf_files[@]} == 0 )); then
  printf 'No GGUF was imported; copied artifacts are available in %s\n' "$DEST"
  exit 0
fi

model_file="$DEST/Modelfile"
selected="${gguf_files[0]}"
for candidate in "${gguf_files[@]}"; do
  if [[ "$(basename "$candidate")" == terra-shield-qwen* ]]; then
    selected="$candidate"
    break
  fi
done
cat > "$model_file" <<EOF
FROM ./$(basename "$selected")
PARAMETER temperature 0.2
EOF

if ! command -v ollama >/dev/null 2>&1; then
  printf 'GGUF copied and Modelfile written; install Ollama, then run: ollama create terra-shield-qwen -f "%s"\n' "$model_file"
  exit 0
fi
ollama create terra-shield-qwen -f "$model_file"
printf 'Registered terra-shield-qwen from %s\n' "$(basename "$selected")"
