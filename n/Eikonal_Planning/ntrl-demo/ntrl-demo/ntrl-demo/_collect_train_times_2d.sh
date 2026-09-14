#!/usr/bin/env bash
# Per-cell training seconds for the 2-D our-method sweep -> ./_train_times_2d.tsv
#
# models/metric's trainer prints no total time (NTFields' does), so the wall clock
# comes from the driver's own "[run ] <cell> on <dev>" / "[ok]   <cell>" lines,
# timestamped by `docker logs -t`.  Re-runnable at any point: cells still training
# simply have no [ok] line yet and are left out.
#
#   bash _collect_train_times_2d.sh [container]
set -u
CONTAINER=${1:-ntrl_ours_2d_train}
OUT=${OUT:-./_train_times_2d.tsv}
mkdir -p "$(dirname "$OUT")"  # Experiments/ is root-owned (written from inside the container), so this lands in the repo root

docker logs -t "$CONTAINER" 2>&1 | awk '
    { ts = $1; sub(/\.[0-9]*Z?$/, "", ts); gsub(/[TZ]/, " ", ts)
      cmd = "date -u -d \"" ts "\" +%s"; cmd | getline epoch; close(cmd) }
    # "<ts> [run ] <cell> on <dev>" -- the space inside "[run ]" makes the cell $4
    /\[run \]/ { start[$4] = epoch }
    # "<ts> [ok]   <cell>"
    /\[ok\]/   { if ($3 in start) printf "%s\t%d\n", $3, epoch - start[$3] }
' > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") cells)"
