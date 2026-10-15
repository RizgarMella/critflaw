#!/usr/bin/env bash
# V10: 20 main seeds and 10 isolated-region seeds, sequentially, one light process at a time; skips seeds done.
cd "$(dirname "$0")/.."
OUT=benchmarks/out/pf_v10.jsonl
for s in $(seq 0 19); do
  grep -q "\"seed\": $s, \"isolated\": false" "$OUT" 2>/dev/null || uv run python benchmarks/pf_v10.py "$s" >/dev/null 2>&1
done
for s in $(seq 100 109); do
  grep -q "\"seed\": $s, \"isolated\": true" "$OUT" 2>/dev/null || uv run python benchmarks/pf_v10.py "$s" isolated >/dev/null 2>&1
done
