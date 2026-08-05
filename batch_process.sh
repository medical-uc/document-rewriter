#!/usr/bin/env bash
set -uo pipefail

mkdir -p ./artifacts/outputs ./artifacts/rewrite_lint ./artifacts/rewrite_debug ./artifacts/rewrite_cache

for f in ./inputs/*.md; do
  [ -e "$f" ] || continue          # skip if the glob matched nothing
  base=$(basename "$f")
  base=${base%.*}                  # strip the extension

  echo "=== $f ==="
  python rewrite_medical_md.py "$f" \
    -o "./artifacts/outputs/$base.md" \
    --lint-report "./artifacts/rewrite_lint/$base.txt" \
    --debug-dir "./artifacts/rewrite_debug/$base" \
    --cache-dir ./artifacts/rewrite_cache \
    -v || echo "FAILED: $f" >&2
done
