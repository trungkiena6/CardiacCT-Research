#!/bin/bash
# A: cham test-250 lan 2 (duy nhat) voi cau hinh da chon; sau do B: tinh chinh ResEnc-L 250 epoch LR 1e-3.
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W=/workspace/coroseg-work
LOG=/workspace/finetune_resencl.log
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results
export nnUNet_compile=false PYTHONPATH=/workspace/coroseg/src
log() { echo "$(date +%F_%T) $*" >> $LOG; }
VALNPZ=$(du -sb "$W/testaccess2/val/resenc_l" "$W/testaccess2/val/unet_ft500" | awk '{s+=$1} END{print s}')
NEED=$((VALNPZ * 6))
FREE=$(df -B1 --output=avail / | tail -1)
log "dia: can ~$((NEED / 1073741824)) GB, trong $((FREE / 1073741824)) GB"
[ "$FREE" -gt "$NEED" ] || {
  log "A-LOI: thieu dia"
  exit 1
}
log "A: test access 2 bat dau"
cd /workspace
if "$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/../imagecas_ensemble_access2.py" test > /workspace/test_access_2_test.log 2>&1; then
  log "A-DONE"
else
  test_status=$?
  log "A-LOI rc=$test_status"
  exit "$test_status"
fi
grep "TEST-2 RESULT" /workspace/test_access_2_test.log >> $LOG || {
  log "A-LOI missing TEST-2 RESULT"
  exit 1
}
log "B: tinh chinh ResEnc-L bat dau"
PRE=$W/nnunet_results/Dataset901_ImageCAS_S1/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres/fold_0/checkpoint_final.pth
if nnUNetv2_train 901 3d_fullres 0 -tr nnUNetTrainer_ft250_NM -p nnUNetResEncUNetL40GPlans -pretrained_weights "$PRE" > "$W/logs/train_resencL_ft250_fold0.log" 2>&1; then
  log "B-DONE rc=0"
else
  train_status=$?
  log "B-LOI rc=$train_status"
  exit "$train_status"
fi
