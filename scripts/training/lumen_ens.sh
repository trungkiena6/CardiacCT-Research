#!/bin/bash
# ImageCAS-X lumen access 2 on one A100:
#   Kaggle volumes + ImageCAS-X labels -> Dataset906 with the run-1 plans (uploaded, not re-planned) -> preprocess once
#   (both plans share nnUNetPlans_3d_fullres) -> validation probabilities of M1 -> train M3 (U-Net) -> M3 validation
#   -> train M2 (ResEnc-L run 2) -> M2 validation. Validation: step 0.25, --save_probabilities, checkpoint_final.
# Test images and labels are not even linked here; lumen_test.sh does that only after lumen_access2_choice.json exists.
# Markers in /workspace/lumen_ens.log: DATA-READY, VAL-M1-DONE, TRAIN-M3-DONE, VAL-M3-DONE, TRAIN-M2-DONE,
#   VAL-PROBS-DONE, LOI <step> on failure. Re-running resumes after the last finished step.
set -u
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W=/workspace/xwork
LOG=/workspace/lumen_ens.log
mkdir -p $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=/workspace/coroseg/src
ID=906
DS=Dataset906_ImageCASX_lumen
PRE=$W/nnunet_preprocessed/$DS
RAW=$W/nnunet_raw/$DS
TR=nnUNetTrainer_250epochs_NoMirroring
TR2=nnUNetTrainer_250epochs_NoMirroring_seed2
PL=nnUNetResEncUNetL40GPlans
PU=nnUNetPlans
log() { echo "$(date +%F_%T) $*" >> $LOG; }
free_disk() { df -h /workspace | tail -1 | awk '{print $4" trong"}'; }

NN=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
F=$NN/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_seed2.py
[ -f $F ] || cat > $F << 'EOF'
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring


class nnUNetTrainer_250epochs_NoMirroring_seed2(nnUNetTrainer_250epochs_NoMirroring):
    """Identical recipe; separate name so the second ResEnc-L run gets its own output folder."""
    pass
EOF

if [ ! -f $W/.data_ready ]; then
  CASES=/workspace/cases
  if [ "$(find $CASES -name "*.img.nii.gz" 2> /dev/null | wc -l)" -lt 1000 ]; then
    SRC=$(KAGGLEHUB_CACHE=/workspace/kagglehub KAGGLE_API_TOKEN=$(cat /workspace/.kaggle_token) python3 -c "import kagglehub; print(kagglehub.dataset_download(\"xiaoweixumedicalai/imagecas\"))" 2>> $LOG | tail -1)
    log "kaggle: $SRC"
    [ -d "$SRC" ] || {
      log "LOI kaggle"
      exit 1
    }
    mkdir -p $CASES
    for g in 1-200 201-400 401-600 601-800 801-1000; do
      mkdir -p $CASES/$g
      [ -e "$SRC/$g.zip" ] || ln -s "$SRC/$g.change2zip" "$SRC/$g.zip"
      7z x -y -o$CASES/$g "$SRC/$g.zip" > /dev/null 2>&1 || {
        log "LOI giai nen $g"
        exit 1
      }
      sub=$(find $CASES/$g -mindepth 1 -maxdepth 1 -type d | head -1)
      if [ -n "$sub" ] && [ "$(find $CASES/$g -maxdepth 1 -name "*.nii.gz" | wc -l)" -eq 0 ]; then mv "$sub"/* $CASES/$g/ && rmdir "$sub"; fi
      rm -f "$SRC/$g.zip" "$SRC/$g.change2zip" "$SRC/$g".z0*
    done
    rm -rf "$SRC"
    log "giai nen: $(find $CASES -name "*.img.nii.gz" | wc -l) ca; $(free_disk)"
  fi
  while [ ! -f /workspace/.upload_done ]; do sleep 30; done
  [ -d /workspace/xdata/ImageCAS-X_dataset/segmentations ] || {
    log "LOI xdata"
    exit 1
  }
  [ -f $PRE/$PL.json ] && [ -f $PRE/$PU.json ] && [ -f $W/splits_run1.json ] || {
    log "LOI plans run 1"
    exit 1
  }
  DATASET_BUILDER=("$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/build_imagecasx_dataset.py")
  "${DATASET_BUILDER[@]}" train-only >> $LOG 2>&1 && "${DATASET_BUILDER[@]}" add-val >> $LOG 2>&1 || {
    log "LOI build"
    exit 1
  }
  cmp -s $PRE/splits_final.json $W/splits_run1.json || python3 -c "import json,sys; a=json.load(open('$PRE/splits_final.json')); b=json.load(open('$W/splits_run1.json')); sys.exit(a!=b)" || {
    log "LOI splits khac run 1"
    exit 1
  }
  nnUNetv2_preprocess -d $ID -plans_name $PU -c 3d_fullres -np 32 > /workspace/pp_lumen.log 2>&1 || {
    log "LOI preprocess"
    exit 1
  }
  python3 - << EOF || {
import json, os
from pathlib import Path
W = Path("$W"); RAW = Path("$RAW")
val = json.load(open("$PRE/splits_final.json"))[0]["val"]
for d in ("val_images", "val_labels"):
    (W / d).mkdir(exist_ok=True)
for c in val:
    for s, t in ((RAW / "imagesTr" / f"{c}_0000.nii.gz", W / "val_images" / f"{c}_0000.nii.gz"), (RAW / "labelsTr" / f"{c}.nii.gz", W / "val_labels" / f"{c}.nii.gz")):
        if not t.exists():
            os.link(s, t)
EOF
    log "LOI val links"
    exit 1
  }
  touch $W/.data_ready
  log "DATA-READY: imagesTr $(ls $RAW/imagesTr | wc -l) val $(ls $W/val_images | wc -l) preprocessed $(ls $PRE/nnUNetPlans_3d_fullres | wc -l) files; $(free_disk)"
fi

pred_val() { # tag trainer plans
  local out=$W/val_prob_$1
  [ -f $out/.done ] && return 0
  nnUNetv2_predict -i $W/val_images -o $out -d $ID -c 3d_fullres -f 0 -tr $2 -p $3 -chk checkpoint_final.pth -step_size 0.25 --save_probabilities -npp 4 -nps 4 > /workspace/pred_val_$1.log 2>&1 || {
    log "LOI predict val $1"
    exit 1
  }
  touch $out/.done
  log "VAL-$1-DONE: $(ls $out/*.npz | wc -l) npz; $(free_disk)"
}
train() { # tag trainer plans
  local R=$W/nnunet_results/$DS/${2}__${3}__3d_fullres/fold_0 C=""
  [ -f $R/checkpoint_final.pth ] && return 0
  [ -f $R/checkpoint_latest.pth ] && C="--c"
  log "TRAIN $1 bat dau $C"
  OMP_NUM_THREADS=2 nnUNetv2_train $ID 3d_fullres 0 -tr $2 -p $3 $C > /workspace/train_$1.log 2>&1 || {
    log "LOI train $1"
    exit 1
  }
  log "TRAIN-$1-DONE"
}

pred_val M1 $TR $PL
train M3 $TR $PU
pred_val M3 $TR $PU
train M2 $TR2 $PL
pred_val M2 $TR2 $PL
log "VAL-PROBS-DONE"
