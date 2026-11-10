#!/bin/bash
# Mask dong mach chu (TotalSegmentator total, fast, CPU) cho 194 ca val/test — dung cho gating giai phau (loai voi goc DMC). Khong dung nhan.
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOTALSEG_HOME_DIR=/workspace/totalseg_home
OUT=/workspace/cacseg-work/aorta
mkdir -p $OUT
TS=/workspace/tsvenv/bin/TotalSegmentator
run() {
  f="$1"
  s=$(basename "$f" _0000.nii.gz)
  [ -f "$OUT/$s/aorta.nii.gz" ] && return 0
  $TS -i "$f" -o "$OUT/$s" -rs aorta --fast -d cpu -ns 4 > "$OUT/$s.log" 2>&1 || echo "LOI $s" >> $OUT/_errors.log
}
export -f run
export OUT TS
ls /workspace/cacseg-work/val_all/*.nii.gz /workspace/cacseg-work/test_all/*.nii.gz | xargs -P 8 -I{} bash -c "run {}"
echo "$(date +%F_%T) AORTA-DONE: $(ls $OUT/*/aorta.nii.gz 2> /dev/null | wc -l)/194, loi: $(cat $OUT/_errors.log 2> /dev/null | wc -l)" >> $OUT/_progress.log
