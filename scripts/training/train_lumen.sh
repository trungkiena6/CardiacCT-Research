#!/bin/bash
# ImageCAS-X lumen benchmark on one A100: Kaggle volumes + Zenodo labels -> Dataset906 (560 train / 80 val / 160 test, their split)
#   -> fingerprint+plans on the 560 -> +val -> preprocess (ResEnc-L 40G plan only) -> train ResEnc-L 250 ep no-mirroring
#   -> predict test-160 (step 0.25, thr 0.5, no component removal) -> score with our evaluator and with ImageCAS-X evaluate.py.
# Markers in /workspace/xbench.log: DATA-READY, TRAIN bat dau, TRAIN-DONE, XBENCH-DONE, LOI <step> on failure.
set -u
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W=/workspace/xwork
LOG=/workspace/xbench.log
mkdir -p $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=/workspace/coroseg/src
ID=906
DS=Dataset906_ImageCASX_lumen
PRE=$W/nnunet_preprocessed/$DS
TR=nnUNetTrainer_250epochs_NoMirroring
PL=nnUNetResEncUNetL40GPlans
log() { echo "$(date +%F_%T) $*" >> $LOG; }
free_disk() { df -h / | tail -1 | awk '{print $4" trong"}'; }

if [ ! -f $PRE/splits_final.json ] || [ ! -f $PRE/$PL.json ]; then
  while pgrep -f "kagglehub.dataset_download" > /dev/null; do sleep 60; done
  SRC=$(KAGGLEHUB_CACHE=/workspace/kagglehub KAGGLE_API_TOKEN=$(cat /workspace/.kaggle_token) python3 -c "import kagglehub; print(kagglehub.dataset_download(\"xiaoweixumedicalai/imagecas\"))" 2>> $LOG | tail -1)
  log "kaggle: $SRC"
  [ -d "$SRC" ] || {
    log "LOI kaggle"
    exit 1
  }
  CASES=/workspace/cases
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
  while ! grep -q POST-DL-DONE /workspace/dl/post_dl.log 2> /dev/null; do sleep 30; done
  [ -d /workspace/xdata/ImageCAS-X_dataset/segmentations ] || {
    log "LOI xdata"
    exit 1
  }
  DATASET_BUILDER=("$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/build_imagecasx_dataset.py")
  "${DATASET_BUILDER[@]}" train-only >> $LOG 2>&1 || {
    log "LOI train-only"
    exit 1
  }
  nnUNetv2_extract_fingerprint -d $ID --verify_dataset_integrity -np 32 > /workspace/pp_x.log 2>&1 || {
    log "LOI fingerprint"
    exit 1
  }
  nnUNetv2_plan_experiment -d $ID >> /workspace/pp_x.log 2>&1 || {
    log "LOI plan"
    exit 1
  }
  nnUNetv2_plan_experiment -d $ID -pl nnUNetPlannerResEncL -gpu_memory_target 40 -overwrite_plans_name $PL >> /workspace/pp_x.log 2>&1 || {
    log "LOI plan ResEncL"
    exit 1
  }
  log "fingerprint+plans tren $(ls $W/nnunet_raw/$DS/imagesTr | wc -l) ca train"
  "${DATASET_BUILDER[@]}" add-val >> $LOG 2>&1 && "${DATASET_BUILDER[@]}" test >> $LOG 2>&1 && "${DATASET_BUILDER[@]}" volumes >> $LOG 2>&1 || {
    log "LOI add-val/test/volumes"
    exit 1
  }
  did=$(python3 -c "import json; print(json.load(open(\"$PRE/$PL.json\"))[\"configurations\"][\"3d_fullres\"][\"data_identifier\"])")
  log "data_identifier ResEncL = $did"
  if [ "$did" = "nnUNetPlans_3d_fullres" ]; then
    nnUNetv2_preprocess -d $ID -c 3d_fullres -np 32 >> /workspace/pp_x.log 2>&1 || {
      log "LOI preprocess"
      exit 1
    }
  else
    nnUNetv2_preprocess -d $ID -plans_name $PL -c 3d_fullres -np 32 >> /workspace/pp_x.log 2>&1 || {
      log "LOI preprocess ResEncL"
      exit 1
    }
  fi
  n=$(python3 -c "import json; print(len(json.load(open(\"$PRE/splits_final.json\"))))")
  [ "$n" = "1" ] || {
    log "LOI splits_final co $n fold"
    exit 1
  }
  log "DATA-READY: imagesTr $(ls $W/nnunet_raw/$DS/imagesTr | wc -l) imagesTs $(ls $W/nnunet_raw/$DS/imagesTs | wc -l) test_labels $(ls /workspace/test_labels_x | wc -l) preprocessed $(ls $PRE/$did | wc -l) files; $(free_disk)"
fi

R=$W/nnunet_results/$DS/${TR}__${PL}__3d_fullres/fold_0
if [ ! -f $R/checkpoint_final.pth ]; then
  log "TRAIN bat dau"
  export OMP_NUM_THREADS=2
  nnUNetv2_train $ID 3d_fullres 0 -tr $TR -p $PL > /workspace/train_x.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  log "TRAIN-DONE"
fi
export OMP_NUM_THREADS=1
if [ ! -f $W/xbench_ours_result.json ]; then
  nnUNetv2_predict -i $W/nnunet_raw/$DS/imagesTs -o $W/test_pred -d $ID -c 3d_fullres -f 0 -tr $TR -p $PL -chk checkpoint_final.pth -step_size 0.25 -npp 4 -nps 4 > /workspace/pred_x.log 2>&1 || {
    log "LOI predict"
    exit 1
  }
  "$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/../score_segmentation.py" $W/test_pred /workspace/test_labels_x $W/xbench_ours_result.json >> $LOG 2>&1 || {
    log "LOI score ours"
    exit 1
  }
fi
export ImageCAS_X_data_path=/workspace/xdata/ImageCAS-X_dataset ImageCAS_X_results_path=/workspace/xresults
mkdir -p /workspace/xresults/coroseg_resencl/predictions
for f in $W/test_pred/*.nii.gz; do [ -e /workspace/xresults/coroseg_resencl/predictions/$(basename $f) ] || ln $f /workspace/xresults/coroseg_resencl/predictions/; done
cd /workspace/ImageCAS-X && python3 -m evaluate -c /workspace/coroseg_resencl.json -r /workspace/xresults/coroseg_resencl -j 32 > /workspace/xeval.log 2>&1 || {
  log "LOI evaluate.py"
  exit 1
}
log "XBENCH-DONE: $(ls /workspace/xresults/coroseg_resencl/*results*.json 2> /dev/null)"
