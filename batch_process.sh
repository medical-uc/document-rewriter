#!/usr/bin/env bash
set -uo pipefail

# Inputs are nested by subject and book (inputs/<subject>/<book>/<chapter>.md),
# and that structure is mirrored into the output and debug trees so chapters
# from different books never collide on a shared basename.

mkdir -p ./artifacts/outputs ./artifacts/rewrite_debug ./artifacts/rewrite_cache

total=0
warned=0
failed=0

# The loop reads from a process substitution rather than the tail of a pipe so
# that the tallies survive it. A piped `while` runs in a subshell, and its
# counters die with it.
while IFS= read -r -d '' f; do
  rel=${f#./inputs/}               # subject/book/chapter.md
  stem=${rel%.*}                   # subject/book/chapter

  echo "=== $rel ==="
  mkdir -p "./artifacts/outputs/$(dirname "$stem")"
  python3 rewrite_medical_md.py "$f" \
    -o "./artifacts/outputs/$stem.md" \
    --debug-dir "./artifacts/rewrite_debug/$stem" \
    --cache-dir ./artifacts/rewrite_cache \
    -v "$@"
  status=$?
  total=$((total + 1))

  # Exit 1 means the chapter was written and verification had something to say
  # about it. Every chapter of a rewrite raises warnings, so reporting that as
  # a failure would mean reporting every run as a failure. Only the codes that
  # leave no document behind count: 2 for input that yielded no content, 3 for
  # an unreachable backend or a rewrite that came back unusable.
  case $status in
    0) ;;
    1) warned=$((warned + 1))
       echo "warnings: $rel" >&2 ;;
    *) failed=$((failed + 1))
       echo "FAILED: $rel (exit $status)" >&2 ;;
  esac
done < <(find ./inputs -type f -name '*.md' -print0 | sort -z)

echo "$total chapter(s): $failed failed, $warned with warnings" >&2
[ "$failed" -eq 0 ]
