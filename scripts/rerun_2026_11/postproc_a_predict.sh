#!/bin/bash
# Post-processing A (PREREG_postproc_A.md): member probability maps on the 50 Split-1 validation cases,
# same configuration as test_access_3_choice.json (checkpoint_final, step 0.25, no TTA). No test data.
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/postproc_a.log
log() { echo "$(date +%F_%T) A $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src COROSEG_WORK=$W
log "PRED start: val_in $(ls $W/val_in | wc -l) cases"
cd $C/scripts && python3 - >> /workspace/logs/postproc_a_pred.log 2>&1 << "PY"
import sys
sys.path.insert(0, ".")
from imagecas_ensemble_access3 import predict, pred_dir, W
for m in ["unet_ft500", "resenc_l", "resenc_l_ft"]:
    out = pred_dir(m, "val", "checkpoint_final.pth", 0.25)
    predict(m, W / "val_in", out, "checkpoint_final.pth", 0.25)
    print("MEMBER-DONE", m, out, flush=True)
PY
rc=$?
if [ $rc -eq 0 ]; then log "PRED-DONE: $(for d in $W/testaccess3/val/*__s0.25; do echo -n "$(basename $d) $(ls $d/*.npz | wc -l) npz; "; done)"; else log "LOI predict rc=$rc"; fi
