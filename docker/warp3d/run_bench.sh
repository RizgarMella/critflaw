#!/bin/bash
# Run one WARP3D example inside the container and collect wall time / peak RSS.
#
#   run_bench.sh <example dir under $WARP3D_HOME> <input file> <threads> <label>
#
# The example is copied to container-local /tmp (not the Windows bind mount) so
# that file I/O does not distort the timing. Text output, the /usr/bin/time -v
# report and a short summary are copied to /out/<label>/ (bind mount).
set -u
ex=$1; inp=$2; thr=$3; label=$4
src="$WARP3D_HOME/$ex"
wd=/tmp/run_$label
rm -rf "$wd"; cp -r "$src" "$wd"; cd "$wd" || exit 1
export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr
ulimit -s 100000 2>/dev/null || ulimit -s unlimited
start=$(date +%s.%N)
/usr/bin/time -v -o time_v.txt "$WARP3D_HOME/run_linux/warp3d.exe" < "$inp" > out.txt 2> err.txt
rc=$?
end=$(date +%s.%N)
mkdir -p "/out/$label"
{
  echo "example=$ex input=$inp threads=$thr exit=$rc"
  echo "wall_seconds=$(echo "$end - $start" | bc 2>/dev/null || python3 -c "print($end-$start)")"
  grep -E "Elapsed \(wall|Maximum resident|User time|System time" time_v.txt
  echo "output_lines=$(wc -l < out.txt)"
} | tee "/out/$label/summary.txt"
cp out.txt err.txt time_v.txt "/out/$label/"
