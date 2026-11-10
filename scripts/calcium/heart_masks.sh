#!/bin/bash
# Mask tim TotalSegmentator (total, fast, CPU) cho 776 anh COCA gated: 1 ca truoc de tai trong so, sau do 24 tien trinh song song.
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOTALSEG_HOME_DIR=/workspace/totalseg_home
OUT=/workspace/cacseg-work/heart
mkdir -p $OUT
TS=/workspace/tsvenv/bin/TotalSegmentator
run() {
  f="$1"
  s=$(basename "$f" _0000.nii.gz)
  [ -f "$OUT/$s/heart.nii.gz" ] && return 0
  $TS -i "$f" -o "$OUT/$s" -rs heart --fast -d cpu -ns 4 > "$OUT/$s.log" 2>&1 || echo "LOI $s" >> $OUT/_errors.log
}
export -f run
export OUT TS
first=$(ls /workspace/cacseg-work/raw_all/images/*.nii.gz | head -1)
run "$first"
echo "$(date +%F_%T) ca dau xong: $(ls $OUT/$(basename $first _0000.nii.gz)/ 2> /dev/null | tr "\n" " ")" >> $OUT/_progress.log
ls /workspace/cacseg-work/raw_all/images/*.nii.gz | xargs -P 10 -I{} bash -c "run {}"
echo "$(date +%F_%T) HEART-DONE: $(ls $OUT/*/heart.nii.gz 2> /dev/null | wc -l)/776, loi: $(cat $OUT/_errors.log 2> /dev/null | wc -l)" >> $OUT/_progress.log
