#!/bin/bash
# Experiment C (PREREG_final_split1.md, 03/11/2026): seed replicate k of the archived ResEnc-L recipe
# (nnUNetTrainer_250epochs_NoMirroring_seed<k>: identical class, new random seed, own results folder) on ImageCAS Split 1.
# Usage: bash expC_seed.sh <2|3>. Markers in /workspace/expC.log: C seed<k> TRAIN start / TRAIN-DONE / LOI.
# Refuses to start while another nnUNetv2_train is running on this GPU unless EXPC_FORCE=1. Resumes with --c.
set -u
K=${1:?seed number}
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/expC.log
log() { echo "$(date +%F_%T) C seed$K $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src
ID=901
DS=Dataset901_ImageCAS_S1
PL=nnUNetResEncUNetL40GPlans
TR=nnUNetTrainer_250epochs_NoMirroring_seed$K
FD=$W/nnunet_results/$DS/${TR}__${PL}__3d_fullres/fold_0
mkdir -p /workspace/logs
[ -f $W/nnunet_preprocessed/$DS/.pilot_data_ready ] || {
  log "LOI preprocessed Split-1 data not ready"
  exit 1
}
if [ "${EXPC_FORCE:-0}" != "1" ] && pgrep -f "/nnUNetv2_train [0-9]+ 3d_fullres" > /dev/null; then
  log "LOI another nnUNetv2_train is running on this GPU; not starting"
  exit 1
fi
if [ ! -f $FD/checkpoint_final.pth ]; then
  extra=""
  [ -f $FD/checkpoint_latest.pth ] && extra="--c"
  log "TRAIN start $extra (trainer $TR, plans $PL)"
  OMP_NUM_THREADS=2 nnUNet_n_proc_DA=16 nnUNetv2_train $ID 3d_fullres 0 -tr $TR -p $PL $extra >> /workspace/logs/expC_seed${K}_train.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  [ -f $FD/checkpoint_final.pth ] || {
    log "LOI train: no checkpoint_final"
    exit 1
  }
  log "TRAIN-DONE: $(grep -o 'EMA pseudo Dice: [0-9.]*' $(ls -t $FD/training_log_* | head -1) | tail -1), sha256 $(sha256sum $FD/checkpoint_final.pth | cut -c1-64)"
else
  log "already trained: $FD/checkpoint_final.pth"
fi
