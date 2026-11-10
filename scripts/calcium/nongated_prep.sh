#!/bin/bash
# Phan C: 214 CT nguc non-gated COCA -> NIfTI (eval_cacseg_nongated prepare) -> mask tim + dong mach chu (TotalSeg, CPU).
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOTALSEG_HOME_DIR=/workspace/totalseg_home
W=/workspace/cacseg-work/nongated
LOG=/workspace/cacseg/nongated_prep.log
mkdir -p $W
cd /workspace/cacseg && python3 eval_cacseg_nongated.py prepare --root /workspace/coca/cocacoronarycalciumandchestcts-2 --out $W >> $LOG 2>&1 || {
  echo "$(date +%F_%T) LOI prepare" >> $LOG
  exit 1
}
echo "$(date +%F_%T) prepare xong: $(ls $W/images/*.nii.gz | wc -l) anh" >> $LOG
OUT=$W/masks
mkdir -p $OUT
TS=/workspace/tsvenv/bin/TotalSegmentator
run() {
  f="$1"
  s=$(basename "$f" _0000.nii.gz)
  [ -f "$OUT/$s/heart.nii.gz" ] && [ -f "$OUT/$s/aorta.nii.gz" ] && return 0
  $TS -i "$f" -o "$OUT/$s" -rs heart aorta --fast -d cpu -ns 4 > "$OUT/$s.log" 2>&1 || echo "LOI $s" >> $OUT/_errors.log
}
export -f run
export OUT TS
ls $W/images/*.nii.gz | xargs -P 8 -I{} bash -c "run {}"
echo "$(date +%F_%T) NONGATED-MASKS-DONE: $(ls $OUT/*/heart.nii.gz 2> /dev/null | wc -l)/214 tim, $(ls $OUT/*/aorta.nii.gz 2> /dev/null | wc -l)/214 DMC, loi $(cat $OUT/_errors.log 2> /dev/null | wc -l)" >> $LOG
