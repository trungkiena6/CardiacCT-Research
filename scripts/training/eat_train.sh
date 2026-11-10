#!/bin/bash
# EAT self-training on the pod: Dataset907_EAT = ImageCAS-X volumes + rule-based EAT pseudo-labels (uploaded as /workspace/eat_labels.tgz
# containing <id>.eat.nii.gz, 1 = EAT). Same 560 / 80 / 160 split as the lumen benchmark so the 160 test cases stay out-of-sample for
# everything. 3d_lowres (≈1.0 x 0.69 x 0.69 mm, patch 96x160x160), nnUNetTrainer_250epochs_NoMirroring, fold 0 only.
# Markers in /workspace/eat.log: EAT-DATA-READY, EAT-TRAIN bat dau, EAT-TRAIN-DONE, EAT-DONE, LOI <step>.
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W="${COROSEG_EAT_WORK:-/workspace/eatwork}"
LOG=/workspace/eat.log
mkdir -p "$W"
if [[ -e "$W/eat_test_result.json" || -L "$W/eat_test_result.json" ]]; then
  echo "Existing EAT summary preserved; select a fresh COROSEG_EAT_WORK" >&2
  exit 1
fi
if [[ -e "$W/test_pred" || -L "$W/test_pred" ]]; then
  [[ -d "$W/test_pred" ]] || {
    echo "Prediction output is not a directory" >&2
    exit 1
  }
  existing_predictions=$(find "$W/test_pred" -mindepth 1 -maxdepth 1 -print -quit) || {
    echo "Cannot inspect prediction output" >&2
    exit 1
  }
  [[ -z "$existing_predictions" ]] || {
    echo "Existing EAT predictions preserved; select a fresh COROSEG_EAT_WORK" >&2
    exit 1
  }
fi
export COROSEG_EAT_TEST_LABELS="${COROSEG_EAT_TEST_LABELS:-/workspace/eat_test_labels}"
EAT_TEST_CASES="${COROSEG_EAT_TEST_CASES:-/workspace/xdata/ImageCAS-X_dataset/filelist/test.txt}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false
ID=907
DS=Dataset907_EAT
PRE=$W/nnunet_preprocessed/$DS
TR=nnUNetTrainer_250epochs_NoMirroring
CFG=3d_lowres
log() { echo "$(date +%F_%T) $*" >> $LOG; }
until [ -f /workspace/eat_labels.tgz ] && [ -f /workspace/eat_labels.done ]; do sleep 120; done
until grep -q DATA-READY /workspace/xbench.log 2> /dev/null; do sleep 120; done
if [ ! -f $PRE/splits_final.json ] || [ ! -d $PRE/nnUNetPlans_$CFG ]; then
  mkdir -p /workspace/eat_labels && tar --no-same-owner -xzf /workspace/eat_labels.tgz -C /workspace/eat_labels || {
    log "LOI tar labels"
    exit 1
  }
  log "labels: $(ls /workspace/eat_labels | grep -c eat.nii.gz)"
  build() {
    "$WORKFLOW_PYTHON" - "$1" << 'EOF'
import json, os, sys
from pathlib import Path
import numpy as np, SimpleITK as sitk
stage = sys.argv[1]
X = Path('/workspace/xdata/ImageCAS-X_dataset/filelist'); L = Path('/workspace/eat_labels')
RAW = Path(os.environ['nnUNet_raw']) / 'Dataset907_EAT'; PRE = Path(os.environ['nnUNet_preprocessed']) / 'Dataset907_EAT'
ids = lambda n: [s for s in (X / f'{n}.txt').read_text().split() if s.strip()]
def img(i):
    b = (int(i) - 1) // 200 * 200; return Path(f'/workspace/cases/{b + 1}-{b + 200}/{i}.img.nii.gz')
def link(s, d):
    d.parent.mkdir(parents=True, exist_ok=True); d.exists() or os.link(s, d)
def lab(i, d):
    if d.exists(): return
    src = next(p for p in (L / f'{i}.eat.nii.gz', *L.rglob(f'{i}.eat.nii.gz')) if p.exists())
    s = sitk.ReadImage(str(src)); o = sitk.GetImageFromArray((sitk.GetArrayFromImage(s) > 0).astype(np.uint8)); o.CopyInformation(s)
    d.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(o, str(d))
if stage == 'train-only':
    for i in ids('train'): link(img(i), RAW / 'imagesTr' / f'{i}_0000.nii.gz'); lab(i, RAW / 'labelsTr' / f'{i}.nii.gz')
    (RAW / 'dataset.json').write_text(json.dumps({'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'eat': 1}, 'numTraining': len(ids('train')), 'file_ending': '.nii.gz'}, indent=1))
elif stage == 'add-val':
    for i in ids('val'): link(img(i), RAW / 'imagesTr' / f'{i}_0000.nii.gz'); lab(i, RAW / 'labelsTr' / f'{i}.nii.gz')
    d = json.loads((RAW / 'dataset.json').read_text()); d['numTraining'] = len(ids('train')) + len(ids('val')); (RAW / 'dataset.json').write_text(json.dumps(d, indent=1))
    PRE.mkdir(parents=True, exist_ok=True); (PRE / 'splits_final.json').write_text(json.dumps([{'train': ids('train'), 'val': ids('val')}]))
elif stage == 'test':
    for i in ids('test'): link(img(i), RAW / 'imagesTs' / f'{i}_0000.nii.gz'); lab(i, Path(os.environ['COROSEG_EAT_TEST_LABELS']) / f'{i}.nii.gz')
print(stage, 'done', flush=True)
EOF
  }
  build train-only >> $LOG 2>&1 || {
    log "LOI train-only"
    exit 1
  }
  nnUNetv2_extract_fingerprint -d $ID --verify_dataset_integrity -np 32 > /workspace/pp_eat.log 2>&1 || {
    log "LOI fingerprint"
    exit 1
  }
  nnUNetv2_plan_experiment -d $ID >> /workspace/pp_eat.log 2>&1 || {
    log "LOI plan"
    exit 1
  }
  python3 -c "import json; c=json.load(open('$PRE/nnUNetPlans.json'))['configurations']; assert '$CFG' in c, list(c); print('$CFG', c['$CFG']['spacing'], c['$CFG']['patch_size'])" >> $LOG 2>&1 || {
    log "LOI khong co $CFG"
    exit 1
  }
  build add-val >> $LOG 2>&1 && build test >> $LOG 2>&1 || {
    log "LOI add-val/test"
    exit 1
  }
  nnUNetv2_preprocess -d $ID -c $CFG -np 32 >> /workspace/pp_eat.log 2>&1 || {
    log "LOI preprocess"
    exit 1
  }
  log "EAT-DATA-READY: imagesTr $(ls $W/nnunet_raw/$DS/imagesTr | wc -l) imagesTs $(ls $W/nnunet_raw/$DS/imagesTs | wc -l) preprocessed $(ls $PRE/nnUNetPlans_$CFG | wc -l) files; $(df -h / | tail -1 | awk '{print $4" trong"}')"
fi
R=$W/nnunet_results/$DS/${TR}__nnUNetPlans__$CFG/fold_0
if [ ! -f $R/checkpoint_final.pth ]; then
  log "EAT-TRAIN bat dau"
  export OMP_NUM_THREADS=2
  nnUNetv2_train $ID $CFG 0 -tr $TR > /workspace/train_eat.log 2>&1 || {
    log "LOI train"
    exit 1
  }
  log "EAT-TRAIN-DONE"
fi
export OMP_NUM_THREADS=1
nnUNetv2_predict -i $W/nnunet_raw/$DS/imagesTs -o $W/test_pred -d $ID -c $CFG -f 0 -tr $TR -chk checkpoint_final.pth -step_size 0.5 --disable_tta -npp 4 -nps 4 > /workspace/pred_eat.log 2>&1 || {
  log "LOI predict"
  exit 1
}
"$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/../score_eat_predictions.py" --pred-dir "$W/test_pred" --gt-dir "$COROSEG_EAT_TEST_LABELS" --cases "$EAT_TEST_CASES" --output "$W/eat_test_result.json" >> $LOG 2>&1 || {
  log "LOI score"
  exit 1
}
log "EAT-DONE"
