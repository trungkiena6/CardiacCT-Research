#!/bin/bash
# Phase 2 (rerun 02/11/2026): Split-1 three-member ensemble exactly as recorded in test_access_3_choice.json
# (unet_ft500 + resenc_l + resenc_l_ft, checkpoint_final, step 0.25, equal weights, T 0). Fifth access to the Split-1 test set.
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/rerun.log
log() { echo "$(date +%F_%T) P2 $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src COROSEG_WORK=$W
mkdir -p $W/nnunet_raw $W/nnunet_preprocessed $W/nnunet_results/Dataset901_ImageCAS_S1 $W/testaccess2/test /workspace/logs
# Inference-only stand-ins for the two fine-tuning trainers whose source was not archived (nnU-Net only needs the class name to load a checkpoint).
NN=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
STUB=$NN/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_rerun_stubs.py
if [ ! -f $STUB ]; then
  cat > $STUB << 'PYSTUB'
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import (
    nnUNetTrainer_250epochs_NoMirroring,
)


class nnUNetTrainer_ft500_NM(nnUNetTrainer_250epochs_NoMirroring):
    """Inference-only stand-in (rerun 02/11/2026): the original fine-tuning trainer source was not archived.
    Only the class name is needed to load the checkpoint; the network comes from plans.json and the saved weights."""


class nnUNetTrainer_ft250_NM(nnUNetTrainer_250epochs_NoMirroring):
    """Inference-only stand-in; see nnUNetTrainer_ft500_NM."""


class nnUNetTrainer_250epochs_NoMirroring_seed2(nnUNetTrainer_250epochs_NoMirroring):
    """Same recipe as the parent; separate name so the second ResEnc-L lumen run (M2) has its own folder."""
PYSTUB
  log "trainer stubs written: $STUB"
fi
R=$W/nnunet_results/Dataset901_ImageCAS_S1
[ -f $R/nnUNetTrainer_ft250_NM__nnUNetResEncUNetL40GPlans__3d_fullres/fold_0/checkpoint_final.pth ] || tar --no-same-owner -xf /workspace/upload/resencL_ckpts.tar -C $R || {
  log "LOI tar resencL"
  exit 1
}
[ -f $R/nnUNetTrainer_ft500_NM__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth ] || cp -r /workspace/upload/ft500/nnUNetTrainer_ft500_NM__nnUNetPlans__3d_fullres $R/ || {
  log "LOI copy ft500"
  exit 1
}
for m in nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres nnUNetTrainer_ft250_NM__nnUNetResEncUNetL40GPlans__3d_fullres nnUNetTrainer_ft500_NM__nnUNetPlans__3d_fullres; do
  log "ckpt $m sha256 $(sha256sum $R/$m/fold_0/checkpoint_final.pth | cut -c1-64)"
done
python3 $C/src/coronary_tree/data/prepare.py --manifest $C/configs/imagecas_v2_split_manifest.csv --cases-dir /workspace/cases --split 1 --dataset-id 901 --nnunet-raw $W/nnunet_raw --nnunet-preprocessed $W/nnunet_preprocessed --work $W --stage test --workers 32 >> $LOG 2>&1 || {
  log "LOI prepare test S1"
  exit 1
}
[ -e $W/test_in ] || ln -s $W/nnunet_raw/Dataset901_ImageCAS_S1/imagesTs $W/test_in
[ -e $W/test_labels ] || ln -s $W/test_labels_s1 $W/test_labels
cp -n $C/results/imagecas/test_access_3_choice.json $W/test_access_3_choice.json
cp -n /workspace/upload/test_access_2_per_case.csv $W/testaccess2/test/final_per_case.csv
log "DATA-READY: test_in $(ls $W/test_in | wc -l), labels $(ls $W/test_labels | wc -l)"
cd $C/scripts && python3 imagecas_ensemble_access3.py test > /workspace/logs/p2_test.log 2>&1 || {
  log "LOI ensemble test (see /workspace/logs/p2_test.log)"
  exit 1
}
log "P2-DONE: $(grep 'TEST-3 RESULT' /workspace/logs/p2_test.log | cut -c1-500)"
