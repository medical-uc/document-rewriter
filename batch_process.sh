#!/usr/bin/env bash
set -uo pipefail

# Inputs are nested by subject and book (inputs/<subject>/<book>/<chapter>.md),
# and that structure is mirrored into the output and debug trees so chapters
# from different books never collide on a shared basename.

mkdir -p ./artifacts/outputs ./artifacts/rewrite_debug ./artifacts/rewrite_cache

find ./inputs -type f -name '*.md' -print0 | sort -z |
while IFS= read -r -d '' f; do
  rel=${f#./inputs/}               # subject/book/chapter.md
  stem=${rel%.*}                   # subject/book/chapter

  echo "=== $rel ==="
  mkdir -p "./artifacts/outputs/$(dirname "$stem")"
  python3 rewrite_medical_md.py "$f" \
    -o "./artifacts/outputs/$stem.md" \
    --debug-dir "./artifacts/rewrite_debug/$stem" \
    --cache-dir ./artifacts/rewrite_cache \
    -v "$@" || echo "FAILED: $rel" >&2
done
