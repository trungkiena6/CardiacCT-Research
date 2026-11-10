#!/bin/bash
# Mot split chinh thuc ImageCAS (2/3/4). Thu tu tiet kiem dia:
#   Kaggle -> giai nen (xoa zip tung nhom) -> raw 700 (hard link) -> fingerprint+plans tren 700 -> +50 val, splits, test, val-dir
#   -> xoa ca giai nen -> preprocess 750 mot lan -> cho GPU -> train ResEnc-L -> xoa preprocessed + imagesTr -> predict test (buoc 0.25) -> cham 1 lan.
set -u
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
S=$1
ID=$2
W=/workspace/coroseg-work
LOG=/workspace/split${S}.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=/workspace/coroseg/src
export KAGGLEHUB_CACHE=/workspace/kagglehub KAGGLE_API_TOKEN=$(cat /workspace/.kaggle_token)
log() { echo "$(date +%F_%T) S$S $*" >> $LOG; }
DS=Dataset${ID}_ImageCAS_S${S}
PRE=$W/nnunet_preprocessed/$DS
if [ ! -f $PRE/splits_final.json ] || [ ! -d $PRE/nnUNetPlans_3d_fullres ]; then
  SRC=$(python3 -c "import kagglehub; print(kagglehub.dataset_download(\"xiaoweixumedicalai/imagecas\"))" 2>> $LOG | tail -1)
  log "kaggle: $SRC"
  [ -d "$SRC" ] || {
    log "LOI kaggle"
    exit 1
  }
  CASES=$W/cases
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
  log "giai nen: $(find $CASES -name "*.img.nii.gz" | wc -l) ca; $(df -h / | tail -1 | awk "{print \$4\" trong\"}")"
  P="python3 /workspace/coroseg/src/coronary_tree/data/prepare.py --manifest /workspace/coroseg/configs/imagecas_v2_split_manifest.csv --cases-dir $CASES --split $S --dataset-id $ID --nnunet-raw $W/nnunet_raw --nnunet-preprocessed $W/nnunet_preprocessed --work $W --excluded $W/metrics/qc/excluded_cases.csv --workers 48"
  $P --stage train-only >> $LOG 2>&1 || {
    log "LOI train-only"
    exit 1
  }
  nnUNetv2_extract_fingerprint -d $ID --verify_dataset_integrity -np 16 > /workspace/pp_s$S.log 2>&1 || {
    log "LOI fingerprint"
    exit 1
  }
  nnUNetv2_plan_experiment -d $ID >> /workspace/pp_s$S.log 2>&1 || {
    log "LOI plan"
    exit 1
  }
  nnUNetv2_plan_experiment -d $ID -pl nnUNetPlannerResEncL -gpu_memory_target 40 -overwrite_plans_name nnUNetResEncUNetL40GPlans >> /workspace/pp_s$S.log 2>&1 || {
    log "LOI plan ResEncL"
    exit 1
  }
  log "fingerprint+plans tren $(ls $W/nnunet_raw/$DS/imagesTr | wc -l) ca train"
  $P --stage add-val >> $LOG 2>&1 && $P --stage splits >> $LOG 2>&1 && $P --stage test >> $LOG 2>&1 && $P --stage val-dir >> $LOG 2>&1 || {
    log "LOI add-val/splits/test/val-dir"
    exit 1
  }
  rm -rf $CASES
  log "raw xong, da xoa ca giai nen; $(df -h / | tail -1 | awk "{print \$4\" trong\"}")"
  did=$(python3 -c "import json; print(json.load(open(\"$PRE/nnUNetResEncUNetL40GPlans.json\"))[\"configurations\"][\"3d_fullres\"][\"data_identifier\"])")
  nnUNetv2_preprocess -d $ID -c 3d_fullres -np 16 >> /workspace/pp_s$S.log 2>&1 || {
    log "LOI preprocess 750"
    exit 1
  }
  [ "$did" = "nnUNetPlans_3d_fullres" ] || {
    log "data_identifier khac ($did) -> preprocess rieng"
    nnUNetv2_preprocess -d $ID -plans_name nnUNetResEncUNetL40GPlans -c 3d_fullres -np 16 >> /workspace/pp_s$S.log 2>&1 || {
      log "LOI preprocess ResEncL"
      exit 1
    }
  }
  n=$(python3 -c "import json; print(len(json.load(open(\"$PRE/splits_final.json\"))))")
  [ "$n" = "1" ] || {
    log "LOI splits_final co $n fold"
    exit 1
  }
  log "DATA-READY: imagesTr $(ls $W/nnunet_raw/$DS/imagesTr | wc -l) imagesTs $(ls $W/nnunet_raw/$DS/imagesTs | wc -l) test_labels $(ls $W/test_labels_s$S | wc -l) preprocessed $(ls $PRE/nnUNetPlans_3d_fullres | wc -l) files; $(df -h / | tail -1 | awk "{print \$4\" trong\"}")"
fi
# pod B: khong co CACSeg, khong cho
while pgrep -f "[n]nUNetv2_train" > /dev/null || pgrep -f "[n]nUNetv2_predict" > /dev/null; do sleep 120; done
R=$W/nnunet_results/$DS/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres/fold_0
if [ ! -f $R/checkpoint_final.pth ]; then
  log "TRAIN bat dau"
  export OMP_NUM_THREADS=2
  nnUNetv2_train $ID 3d_fullres 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans > /workspace/train_s$S.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  log "TRAIN-DONE"
fi
rm -rf $PRE/nnUNetPlans_3d_fullres $W/nnunet_raw/$DS/imagesTr $W/nnunet_raw/$DS/labelsTr
export OMP_NUM_THREADS=1
nnUNetv2_predict -i $W/nnunet_raw/$DS/imagesTs -o $W/split$S/test_pred -d $ID -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans -chk checkpoint_final.pth -step_size 0.25 -npp 4 -nps 4 > /workspace/pred_s$S.log 2>&1 || {
  log "LOI predict"
  exit 1
}
"$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/../score_segmentation.py" $W/split$S/test_pred $W/test_labels_s$S $W/split${S}_result.json >> $LOG 2>&1 || {
  log "LOI score"
  exit 1
}
rm -rf $W/nnunet_raw/$DS/imagesTs
log "SPLIT-DONE"
