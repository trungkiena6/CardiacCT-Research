#!/bin/bash
# C4: crop (911) voi preset ResEnc-L (patch 40x384x448), NoMirroring 250 ep, sau khi C3 xong.
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/workspace/cacseg-work
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false
LOG=/workspace/cacseg/train_c4.log
until grep -qE "C3-DONE|C3-LOI" /workspace/cacseg/train_c3.log 2> /dev/null; do sleep 120; done
while pgrep -f "[n]nUNetv2_train" > /dev/null; do sleep 60; done
n=$(python3 -c "import json; print(len(json.load(open(\"$W/nnunet_preprocessed/Dataset911_COCA_CAC_HO_CROP/splits_final.json\"))))")
[ "$n" = "1" ] || {
  echo "LOI: splits_final co $n fold" >> $LOG
  exit 1
}
echo "$(date +%F_%T) C4 bat dau" >> $LOG
nnUNetv2_train 911 3d_fullres 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans --npz >> $LOG 2>&1 && echo "$(date +%F_%T) C4-DONE" >> $LOG || echo "$(date +%F_%T) C4-LOI rc=$?" >> $LOG
