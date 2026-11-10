#!/bin/bash
# Experiment B (PREREG_expB_skelrec.md, 03/11/2026): ResEnc-L, archived 250-epoch no-mirroring recipe + Skeleton Recall
# loss (w = 1), ImageCAS Split 1, judged on the 50 validation cases against pilot1000/base_val (val Dice 0.8390).
# No test data. Markers in /workspace/expB.log: TRAIN start / TRAIN-DONE / VAL-DONE / DELTA / B-DONE / LOI.
# Re-running resumes after the last finished step (training resumes from checkpoint_latest with --c).
# Refuses to start while another nnUNetv2_train is running on this GPU (the pilot) unless EXPB_FORCE=1.
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/expB.log
log() { echo "$(date +%F_%T) B $*" >> $LOG; }
summ() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print({k: round(v['mean'],4) for k,v in r.items() if isinstance(v,dict)})" $1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false
export PYTHONPATH=/workspace/rerun/skelrec:$C/src nnUNet_extTrainer=/workspace/rerun/skelrec/skeleton_recall/training/nnUNetTrainer
export SKELREC_WEIGHT=1
ID=901
DS=Dataset901_ImageCAS_S1
PL=nnUNetResEncUNetL40GPlans
TR=nnUNetTrainer_250epochs_NoMirroring_SkelRec
R=$W/nnunet_results/$DS
FD=$R/${TR}__${PL}__3d_fullres/fold_0
OUT=$W/expB
mkdir -p $OUT /workspace/logs
BASE=$W/pilot1000/base_val_result.json
[ -f $BASE ] || {
  log "LOI baseline missing: $BASE"
  exit 1
}
[ -f $W/nnunet_preprocessed/$DS/.pilot_data_ready ] || {
  log "LOI preprocessed Split-1 data not ready"
  exit 1
}
if [ "${EXPB_FORCE:-0}" != "1" ] && pgrep -f "/nnUNetv2_train [0-9]+ 3d_fullres" > /dev/null; then
  log "LOI another nnUNetv2_train is running on this GPU; not starting (set EXPB_FORCE=1 to override)"
  exit 1
fi
log "start: trainer $TR plans $PL SKELREC_WEIGHT=$SKELREC_WEIGHT; skelrec package sha256 $(cat $(find /workspace/rerun/skelrec -name '*.py' | sort) | sha256sum | cut -c1-16)"

# 1. training
if [ ! -f $FD/checkpoint_final.pth ]; then
  extra=""
  [ -f $FD/checkpoint_latest.pth ] && extra="--c"
  log "TRAIN start $extra"
  OMP_NUM_THREADS=2 nnUNet_n_proc_DA=16 nnUNetv2_train $ID 3d_fullres 0 -tr $TR -p $PL $extra >> /workspace/logs/expB_train.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  [ -f $FD/checkpoint_final.pth ] || {
    log "LOI train: no checkpoint_final"
    exit 1
  }
  log "TRAIN-DONE: $(grep -o 'EMA pseudo Dice: [0-9.]*' $(ls -t $FD/training_log_* | head -1) | tail -1), sha256 $(sha256sum $FD/checkpoint_final.pth | cut -c1-64)"
fi

# 2. validation inference, identical to pilot1000/base_val (step 0.5, no TTA)
if [ ! -f $OUT/val_result.json ]; then
  if [ ! -f $OUT/val/.done ]; then
    nnUNetv2_predict -i $W/val_s1 -o $OUT/val -d $ID -c 3d_fullres -f 0 -tr $TR -p $PL -chk checkpoint_final.pth -step_size 0.5 --disable_tta --continue_prediction -npp 4 -nps 4 >> /workspace/logs/expB_pred.log 2>&1 || {
      log "LOI predict"
      exit 1
    }
    [ "$(ls $OUT/val/*.nii.gz | wc -l)" -eq 50 ] || {
      log "LOI predict: incomplete"
      exit 1
    }
    touch $OUT/val/.done
  fi
  python3 $C/scripts/score_segmentation.py $OUT/val $W/val_labels_s1 $OUT/val_result.json --workers 32 >> $LOG 2>&1 || {
    log "LOI score"
    exit 1
  }
  log "VAL-DONE: $(summ $OUT/val_result.json)"
fi

# 3. pre-registered decision
python3 - "$OUT" "$BASE" "$W/pilot1000/base_val_result.per_case.csv" >> $LOG 2>&1 << 'PYDELTA'
import csv, json, sys
import numpy as np
out, base_path, base_csv = sys.argv[1:4]
b = json.load(open(base_path)); p = json.load(open(f"{out}/val_result.json"))
d = {k: round(p[k]["mean"] - b[k]["mean"], 4) for k in b if isinstance(b[k], dict) and k in p}
rb = {r["case"]: float(r["dice"]) for r in csv.DictReader(open(base_csv, encoding="utf-8-sig"))}
rp = {r["case"]: float(r["dice"]) for r in csv.DictReader(open(f"{out}/val_result.per_case.csv", encoding="utf-8-sig"))}
v = np.array([rp[c] - rb[c] for c in sorted(rb)]); rng = np.random.default_rng(0)
bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)]); ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
cov = d.get("coverage_1mm", 0.0)
verdict = "ADOPT (dice >= +0.004 and coverage >= 0)" if d["dice"] >= 0.004 and cov >= 0 else ("NOT ADOPTED; candidate 4th member (0 <= dice < +0.004)" if d["dice"] >= 0 else "DISCARD (dice < 0)")
print("DELTA SkelRec - base on val-50:", d, "paired CI95 dice", [round(x, 4) for x in ci], "->", verdict)
json.dump(dict(base=b, skelrec=p, delta=d, delta_dice_ci95=ci, rule="adopt if dice >= +0.004 and coverage_1mm >= 0", verdict=verdict), open(f"{out}/decision.json", "w"), indent=1)
PYDELTA
log "B-DONE"
