#!/bin/bash
# Phase 1 (rerun 02/11/2026): M1 lumen ResEnc-L on ImageCAS-X test-160, raw (thr 0.5) and with the recorded post-processing
# (lumen_access2_choice.json: hysteresis L 0.3, component filter T 10 mm3). Scored with the repo evaluator and ImageCAS-X evaluate.py.
set -u
C=/workspace/coroseg
W=/workspace/xwork
LOG=/workspace/rerun.log
log() { echo "$(date +%F_%T) P1 $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src
mkdir -p $W/nnunet_raw $W/nnunet_preprocessed $W/nnunet_results /workspace/xdata /workspace/logs
X=/workspace/xdata/ImageCAS-X_dataset
if [ ! -d $X/segmentations ]; then
  unzip -q /workspace/upload/ImageCAS-X_dataset.zip -d /workspace/xdata || {
    log "LOI unzip"
    exit 1
  }
fi
log "xdata: $(ls $X/segmentations | wc -l) segmentations, $(wc -w < $X/filelist/test.txt) test ids"
D=$nnUNet_results/Dataset906_ImageCASX_lumen
[ -d $D ] || cp -r /workspace/upload/lumen/Dataset906_ImageCASX_lumen $D
CK=$D/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres/fold_0/checkpoint_final.pth
log "M1 checkpoint sha256 $(sha256sum $CK | cut -c1-64) ($(stat -c %s $CK) bytes)"
python3 $C/scripts/training/build_imagecasx_dataset.py test >> $LOG 2>&1 && python3 $C/scripts/training/build_imagecasx_dataset.py volumes >> $LOG 2>&1 || {
  log "LOI build test"
  exit 1
}
log "DATA-READY: imagesTs $(ls $nnUNet_raw/Dataset906_ImageCASX_lumen/imagesTs | wc -l), labels $(ls /workspace/test_labels_x | wc -l)"
OUT=$W/test_prob_M1
if [ ! -f $OUT/.done ]; then
  nnUNetv2_predict -i $nnUNet_raw/Dataset906_ImageCASX_lumen/imagesTs -o $OUT -d 906 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans -chk checkpoint_final.pth -step_size 0.25 --save_probabilities -npp 4 -nps 4 > /workspace/logs/p1_pred_M1.log 2>&1 || {
    log "LOI predict M1"
    exit 1
  }
  touch $OUT/.done
fi
log "PREDICT-DONE: $(ls $OUT/*.nii.gz | wc -l) masks, $(ls $OUT/*.npz | wc -l) npz"
python3 $C/scripts/training/lumen_select.py apply --choice $C/results/imagecas-x/lumen_access2_choice.json --prob-root $W --prefix test_prob --out $W/test_final --workers 24 >> $LOG 2>&1 || {
  log "LOI apply"
  exit 1
}
log "APPLY-DONE: $(ls $W/test_final/*.nii.gz | wc -l) masks"
mkdir -p $W/m1_masks
for f in $OUT/*.nii.gz; do ln -f $f $W/m1_masks/$(basename $f); done
python3 $C/scripts/score_segmentation.py $W/m1_masks /workspace/test_labels_x $W/rerun_m1_result.json --workers 32 >> $LOG 2>&1 || {
  log "LOI score M1"
  exit 1
}
python3 $C/scripts/score_segmentation.py $W/test_final /workspace/test_labels_x $W/rerun_m1pp_result.json --workers 32 >> $LOG 2>&1 || {
  log "LOI score PP"
  exit 1
}
summ() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print({k: round(v['mean'],4) for k,v in r.items() if isinstance(v,dict)})" $1; }
log "SCORE-DONE: M1 $(summ $W/rerun_m1_result.json) | M1+PP $(summ $W/rerun_m1pp_result.json)"
if [ ! -d /workspace/ImageCAS-X ]; then
  git clone -q https://github.com/kitbransby/ImageCAS-X /workspace/ImageCAS-X >> $LOG 2>&1 && pip install --break-system-packages -q -e /workspace/ImageCAS-X >> $LOG 2>&1 || log "LOI ImageCAS-X clone/install"
fi
export ImageCAS_X_data_path=$X ImageCAS_X_results_path=/workspace/xresults
for tag in m1:$W/m1_masks m1pp:$W/test_final; do
  name=${tag%%:*}
  src=${tag#*:}
  R=/workspace/xresults/rerun_$name
  mkdir -p $R/predictions
  for f in $src/*.nii.gz; do ln -f $f $R/predictions/$(basename $f); done
  python3 -c "import json; d=json.load(open('$C/scripts/training/coroseg_resencl.json')); p=json.load(open('/workspace/ImageCAS-X/configs/pipeline.json')); d.update(method_name='rerun_$name', data=p['data'], evaluation=p['evaluation']); json.dump(d, open('/workspace/ImageCAS-X/configs/rerun_$name.json', 'w'), indent=1)"
  (cd /workspace/ImageCAS-X && python3 -m evaluate -c /workspace/ImageCAS-X/configs/rerun_$name.json -r $R -j 32 > /workspace/logs/p1_xeval_$name.log 2>&1) && log "XEVAL-DONE $name: $(find /workspace/xresults -name "rerun_${name}_results.json" | head -1)" || log "LOI evaluate.py $name"
done
log "P1-DONE"
