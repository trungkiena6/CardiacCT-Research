#!/bin/bash
# A100: ResEnc-L patch lon (plans nnUNetResEncUNetL40GPlans), fold 0 = 700 train / 50 val khoa.
# Diem dung da thong nhat: Dice val-50 >= 0.835 moi di tiep.
set -euo pipefail
W=/workspace/coroseg-work
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results
export nnUNet_compile=false
export nnUNet_n_proc_DA=24
P=$nnUNet_preprocessed/Dataset901_ImageCAS_S1
m=$(ls "$P/nnUNetPlans_3d_fullres"/*.b2nd 2> /dev/null | grep -v _seg | wc -l)
s=$(ls "$P/nnUNetPlans_3d_fullres"/*_seg.b2nd 2> /dev/null | wc -l)
g=$(ls "$P/gt_segmentations" 2> /dev/null | wc -l)
n=$(python3 -c "import json; print(len(json.load(open('$P/splits_final.json'))))")
echo "data $m | seg $s | gt $g | so fold trong splits: $n"
[ "$m" -ge 750 ] && [ "$s" -ge 750 ] && [ "$g" -ge 750 ] || {
  echo "THIEU DU LIEU"
  exit 1
}
[ "$n" = 1 ] || {
  echo "SPLITS KHONG PHAI 1 FOLD — dung lai kiem tra"
  exit 1
}
[ -f "$P/nnUNetResEncUNetL40GPlans.json" ] || {
  echo "THIEU PLANS"
  exit 1
}
cd "$W"
echo "== A100 RE-L fold0 start $(date +%F_%T) =="
if nnUNetv2_train 901 3d_fullres 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans \
  > "$W/logs/train_resencL_fold0.log" 2>&1; then
  echo "== A100 RE-L fold0 done rc=0 $(date +%F_%T) =="
else
  train_status=$?
  echo "== A100 RE-L fold0 failed rc=$train_status $(date +%F_%T) ==" >&2
  exit "$train_status"
fi
