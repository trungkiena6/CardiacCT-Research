#!/bin/bash
# ImageCAS-X lumen test access 2. Runs ONLY when
# /workspace/xwork/lumen_access2_choice.json exists and says go: links the 160 test scans, predicts them with the
# chosen models (step 0.25, probabilities), applies the chosen L and T, scores with our evaluator and the ImageCAS-X
# evaluate.py, and computes the paired difference to access 1. Nothing is re-chosen here.
# Markers in /workspace/lumen_ens.log: TEST-ACCESS-2 bat dau, TEST-DONE, NO-GO, LOI <step>.
set -u
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W=/workspace/xwork
LOG=/workspace/lumen_ens.log
CH=$W/lumen_access2_choice.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=/workspace/coroseg/src
ID=906
DS=Dataset906_ImageCASX_lumen
RAW=$W/nnunet_raw/$DS
log() { echo "$(date +%F_%T) $*" >> $LOG; }

[ -f $CH ] || {
  log "LOI chua co $CH"
  exit 1
}
GO=$(python3 -c "import json; print(json.load(open('$CH'))['go'])")
[ "$GO" = "True" ] || {
  log "NO-GO: test khong duoc cham, access 1 giu nguyen"
  exit 0
}
CFG=$(python3 -c "import json; print(' '.join(json.load(open('$CH'))['config'].split('+')))")
log "TEST-ACCESS-2 bat dau: choice sha256 $(sha256sum $CH | cut -c1-64), models $CFG"

"$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/build_imagecasx_dataset.py" test >> $LOG 2>&1 && "$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/build_imagecasx_dataset.py" volumes >> $LOG 2>&1 || {
  log "LOI build test"
  exit 1
}
for M in $CFG; do
  case $M in
    M1)
      TR=nnUNetTrainer_250epochs_NoMirroring
      PL=nnUNetResEncUNetL40GPlans
      ;;
    M2)
      TR=nnUNetTrainer_250epochs_NoMirroring_seed2
      PL=nnUNetResEncUNetL40GPlans
      ;;
    M3)
      TR=nnUNetTrainer_250epochs_NoMirroring
      PL=nnUNetPlans
      ;;
  esac
  out=$W/test_prob_$M
  [ -f $out/.done ] && continue
  nnUNetv2_predict -i $RAW/imagesTs -o $out -d $ID -c 3d_fullres -f 0 -tr $TR -p $PL -chk checkpoint_final.pth -step_size 0.25 --save_probabilities -npp 4 -nps 4 > /workspace/pred_test_$M.log 2>&1 || {
    log "LOI predict test $M"
    exit 1
  }
  touch $out/.done
done
python3 /workspace/lumen_select.py apply --choice $CH --prob-root $W --prefix test_prob --out $W/test_final >> $LOG 2>&1 || {
  log "LOI apply"
  exit 1
}
"$WORKFLOW_PYTHON" "$TRAINING_SCRIPT_DIR/../score_segmentation.py" $W/test_final /workspace/test_labels_x $W/lumen_access2_test_result.json >> $LOG 2>&1 || {
  log "LOI score"
  exit 1
}

R=/workspace/xresults/coroseg_lumen_access2
mkdir -p $R/predictions
for f in $W/test_final/*.nii.gz; do [ -e $R/predictions/$(basename $f) ] || ln $f $R/predictions/; done
python3 -c "import json; d=json.load(open('/workspace/coroseg_resencl.json')); p=json.load(open('/workspace/ImageCAS-X/configs/pipeline.json')); d.update(method_name='coroseg_lumen_access2', data=p['data'], evaluation=p['evaluation']); json.dump(d, open('/workspace/coroseg_lumen_access2.json', 'w'), indent=1)"
export ImageCAS_X_data_path=/workspace/xdata/ImageCAS-X_dataset ImageCAS_X_results_path=/workspace/xresults
(cd /workspace/ImageCAS-X && python3 -m evaluate -c /workspace/coroseg_lumen_access2.json -r $R -j 32 > /workspace/xeval_access2.log 2>&1) || {
  log "LOI evaluate.py"
  exit 1
}

python3 - << EOF >> $LOG 2>&1 || {
import csv, json
import numpy as np
a1 = {r["case"]: r for r in csv.DictReader(open("/workspace/xbench_access1_per_case.csv"))}
a2 = {r["case"]: r for r in csv.DictReader(open("$W/lumen_access2_test_result.per_case.csv"))}
cases = sorted(set(a1) & set(a2)); assert len(cases) == 160, len(cases)
f = lambda rows, k: np.array([float(rows[c][k]) for c in cases])
betti = lambda rows: np.abs(f(rows, "n_components_pred") - f(rows, "n_components_gt"))
rng, out = np.random.default_rng(0), {"n": len(cases)}
for k, d in {"dice": f(a2, "dice") - f(a1, "dice"), "cldice": f(a2, "cldice") - f(a1, "cldice"), "betti0_err": betti(a2) - betti(a1)}.items():
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    out[k] = {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
json.dump(out, open("$W/lumen_access2_vs_access1.json", "w"), indent=1); print("PAIRED", out)
EOF
  log "LOI paired"
  exit 1
}
log "TEST-DONE: $(python3 -c "import json; r=json.load(open('$W/lumen_access2_test_result.json')); print({k: round(v['mean'], 4) for k, v in r.items() if isinstance(v, dict)})")"
