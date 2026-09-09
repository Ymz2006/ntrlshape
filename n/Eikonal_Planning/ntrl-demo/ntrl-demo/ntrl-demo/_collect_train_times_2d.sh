#!/usr/bin/env bash
# Per-cell training seconds for the 2-D our-method sweep -> Experiments/3dshape_2d/train_times.tsv
#
# models/metric's trainer prints no total time (NTFields' does), so the wall clock
# comes from the driver's own "[run ] <cell> on <dev>" / "[ok]   <cell>" lines,
# timestamped by `docker logs -t`.  Re-runnable at any point: cells still training
# simply have no [ok] line yet and are left out.
#
#   bash _collect_train_times_2d.sh [container]
set -u
CONTAINER=${1:-ntrl_ours_2d_train}
OUT=Experiments/3dshape_2d/train_times.tsv
mkdir -p "$(dirname "$OUT")"

docker logs -t "$CONTAINER" 2>&1 | awk '
    { ts = $1; sub(/\.[0-9]*Z?$/, "", ts); gsub(/[TZ]/, " ", ts)
      cmd = "date -u -d \"" ts "\" +%s"; cmd | getline epoch; close(cmd) }
    /\[run \]/ { start[$3] = epoch }
    /\[ok\]/   { if ($2 in start) printf "%s\t%d\n", $2, epoch - start[$2] }
' > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") cells)"
