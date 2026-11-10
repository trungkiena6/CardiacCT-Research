#!/bin/bash
# Pilot (PREREG_pilot1000.md, 02/11/2026): ResEnc-L trained for 1000 epochs on ImageCAS Split 1 (700/50), judged on the
# 50 validation cases only, against the archived 250-epoch ResEnc-L scored in the same way. No test set is touched.
# Markers in /workspace/pilot.log: DATA-READY, BASE-VAL-DONE, TRAIN start, TRAIN-DONE, VAL-DONE, DELTA, LOI <step>.
# Re-running resumes after the last finished step (training resumes from checkpoint_latest with --c).
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/pilot.log
log() { echo "$(date +%F_%T) PILOT $*" >> $LOG; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
summ() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print({k: round(v['mean'],4) for k,v in r.items() if isinstance(v,dict)})" $1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src
S=1
ID=901
DS=Dataset901_ImageCAS_S1
PRE=$W/nnunet_preprocessed/$DS
RAW=$W/nnunet_raw/$DS
R=$W/nnunet_results/$DS
TRB=nnUNetTrainer_250epochs_NoMirroring
TRP=nnUNetTrainer_1000epochs_NoMirroring
PL=nnUNetResEncUNetL40GPlans
RESL=$R/${TRB}__${PL}__3d_fullres/fold_0/checkpoint_final.pth
OUT=$W/pilot1000
mkdir -p $OUT /workspace/logs
P="python3 $C/src/coronary_tree/data/prepare.py --manifest $C/configs/imagecas_v2_split_manifest.csv --cases-dir /workspace/cases --split $S --dataset-id $ID --nnunet-raw $W/nnunet_raw --nnunet-preprocessed $W/nnunet_preprocessed --work $W --excluded /workspace/rerun/excluded_cases.csv --workers 32"
[ -f $RESL ] || {
  log "LOI archived ResEnc-L missing: $RESL"
  exit 1
}
log "start; archived ResEnc-L sha256 $(sha256sum $RESL | cut -c1-64); $(free_gb) GB free"

# 1. data: 700 train -> fingerprint + plans -> archived ResEnc-L plans -> +50 val -> splits -> val-dir -> preprocess once
if [ ! -f $PRE/.pilot_data_ready ]; then
  while [ "$(free_gb)" -lt 130 ]; do
    log "waiting for disk ($(free_gb) GB free, need 130)"
    sleep 600
  done
  [ -d $RAW/imagesTr ] && [ "$(ls $RAW/imagesTr | wc -l)" -ge 700 ] || $P --stage train-only >> $LOG 2>&1 || {
    log "LOI train-only"
    exit 1
  }
  if [ ! -f $PRE/$PL.json ]; then
    nnUNetv2_extract_fingerprint -d $ID --verify_dataset_integrity -np 16 > /workspace/logs/pilot_pp.log 2>&1 || {
      log "LOI fingerprint"
      exit 1
    }
    nnUNetv2_plan_experiment -d $ID >> /workspace/logs/pilot_pp.log 2>&1 || {
      log "LOI plan"
      exit 1
    }
    nnUNetv2_plan_experiment -d $ID -pl nnUNetPlannerResEncL -gpu_memory_target 40 -overwrite_plans_name $PL >> /workspace/logs/pilot_pp.log 2>&1 || {
      log "LOI plan ResEncL"
      exit 1
    }
    python3 - "$PRE/$PL.json" "$R/${TRB}__${PL}__3d_fullres/plans.json" >> $LOG 2>&1 << 'PYPLANS'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
a, b = new["configurations"]["3d_fullres"], old["configurations"]["3d_fullres"]
diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print("ResEnc-L plans re-planned vs archived: 3d_fullres differs in", diff or "nothing")
PYPLANS
    cp -f $R/${TRB}__${PL}__3d_fullres/plans.json $PRE/$PL.json
    log "fingerprint+plans on $(ls $RAW/imagesTr | wc -l) train cases; archived ResEnc-L plans copied to $PRE/$PL.json"
  fi
  [ "$(ls $RAW/imagesTr | wc -l)" -ge 750 ] || $P --stage add-val >> $LOG 2>&1 || {
    log "LOI add-val"
    exit 1
  }
  $P --stage splits >> $LOG 2>&1 || {
    log "LOI splits"
    exit 1
  }
  [ -d $W/val_s$S ] && [ "$(ls $W/val_s$S | wc -l)" -eq 50 ] || $P --stage val-dir >> $LOG 2>&1 || {
    log "LOI val-dir"
    exit 1
  }
  did=$(python3 -c "import json; print(json.load(open('$PRE/$PL.json'))['configurations']['3d_fullres']['data_identifier'])")
  nnUNetv2_preprocess -d $ID -c 3d_fullres -np 16 >> /workspace/logs/pilot_pp.log 2>&1 || {
    log "LOI preprocess"
    exit 1
  }
  [ "$did" = "nnUNetPlans_3d_fullres" ] || {
    log "data_identifier $did differs -> separate preprocessing"
    nnUNetv2_preprocess -d $ID -plans_name $PL -c 3d_fullres -np 16 >> /workspace/logs/pilot_pp.log 2>&1 || {
      log "LOI preprocess ResEncL"
      exit 1
    }
  }
  n=$(python3 -c "import json; print(len(json.load(open('$PRE/splits_final.json'))))")
  [ "$n" = "1" ] || {
    log "LOI splits_final has $n folds"
    exit 1
  }
  m=$(ls $PRE/nnUNetPlans_3d_fullres/*.b2nd | grep -v _seg | wc -l)
  [ "$m" -ge 750 ] || {
    log "LOI preprocessed $m < 750"
    exit 1
  }
  cp -f $W/dataset_build_s$S.json $OUT/dataset_build_s$S.pilot.json
  touch $PRE/.pilot_data_ready
  log "DATA-READY: imagesTr $(ls $RAW/imagesTr | wc -l), preprocessed $m, val_s$S $(ls $W/val_s$S | wc -l), $(free_gb) GB free"
fi

predict() { # tag trainer out
  [ -f $3/.done ] && return 0
  nnUNetv2_predict -i $W/val_s$S -o $3 -d $ID -c 3d_fullres -f 0 -tr $2 -p $PL -chk checkpoint_final.pth -step_size 0.5 --disable_tta --continue_prediction -npp 4 -nps 4 >> /workspace/logs/pilot_pred_$1.log 2>&1 || {
    log "LOI predict $1"
    return 1
  }
  [ "$(ls $3/*.nii.gz | wc -l)" -eq 50 ] || {
    log "LOI predict $1: incomplete"
    return 1
  }
  touch $3/.done
}

# 2. baseline: the archived 250-epoch ResEnc-L on the 50 validation cases (step 0.5, no TTA), same scorer
if [ ! -f $OUT/base_val_result.json ]; then
  predict base $TRB $OUT/base_val || exit 1
  python3 $C/scripts/score_segmentation.py $OUT/base_val $W/val_labels_s$S $OUT/base_val_result.json --workers 32 >> $LOG 2>&1 || {
    log "LOI score base"
    exit 1
  }
  log "BASE-VAL-DONE: $(summ $OUT/base_val_result.json) (recorded access-2 value 0.8390)"
fi

# 3. training: 1000 epochs, no mirroring, archived plans
FD=$R/${TRP}__${PL}__3d_fullres/fold_0
if [ ! -f $FD/checkpoint_final.pth ]; then
  extra=""
  [ -f $FD/checkpoint_latest.pth ] && extra="--c"
  log "TRAIN start $extra"
  OMP_NUM_THREADS=2 nnUNet_n_proc_DA=16 nnUNetv2_train $ID 3d_fullres 0 -tr $TRP -p $PL $extra >> /workspace/logs/pilot_train.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  [ -f $FD/checkpoint_final.pth ] || {
    log "LOI train: no checkpoint_final"
    exit 1
  }
  log "TRAIN-DONE: $(grep -o 'EMA pseudo Dice: [0-9.]*' $(ls -t $FD/training_log_* | head -1) | tail -1), best EMA $(python3 -c "import torch; print(round(float(torch.load('$FD/checkpoint_final.pth', map_location='cpu', weights_only=False)['_best_ema']),4))"), sha256 $(sha256sum $FD/checkpoint_final.pth | cut -c1-64)"
fi

# 4. the 1000-epoch model on the same 50 validation cases, same inference configuration
if [ ! -f $OUT/pilot_val_result.json ]; then
  predict pilot $TRP $OUT/pilot_val || exit 1
  python3 $C/scripts/score_segmentation.py $OUT/pilot_val $W/val_labels_s$S $OUT/pilot_val_result.json --workers 32 >> $LOG 2>&1 || {
    log "LOI score pilot"
    exit 1
  }
  log "VAL-DONE: $(summ $OUT/pilot_val_result.json)"
fi

# 5. pre-registered decision (PREREG_pilot1000.md): go if the validation Dice gain is >= 0.006
python3 - "$OUT" >> $LOG 2>&1 << 'PYDELTA'
import json, sys
out = sys.argv[1]
b = json.load(open(f"{out}/base_val_result.json")); p = json.load(open(f"{out}/pilot_val_result.json"))
d = {k: round(p[k]["mean"] - b[k]["mean"], 4) for k in b if isinstance(b[k], dict) and k in p}
verdict = "GO (>= +0.006)" if d["dice"] >= 0.006 else "NO-GO (< +0.006)"
print("DELTA 1000ep - 250ep on val-50:", d, "->", verdict)
json.dump(dict(base=b, pilot=p, delta=d, rule="dice gain >= 0.006", verdict=verdict), open(f"{out}/pilot_decision.json", "w"), indent=1)
PYDELTA
log "PILOT-DONE"
