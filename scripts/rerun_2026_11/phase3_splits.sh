#!/bin/bash
# Phase 3 (rerun 02/11/2026): single ResEnc-L of each official ImageCAS split on its 250 test cases (step 0.25, thr 0.5, no component removal).
# Split 1 reuses the resenc_l member prediction of phase 2, as the recorded split-1 result did.
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/rerun.log
log() { echo "$(date +%F_%T) P3 $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false PYTHONPATH=$C/src
mkdir -p $W/nnunet_raw $W/nnunet_preprocessed $W/nnunet_results /workspace/logs
R=$W/nnunet_results
[ -d $R/Dataset902_ImageCAS_S2 ] || tar --no-same-owner -xf /workspace/upload/split2_resencl_ckpt.tar -C $R || {
  log "LOI tar split2"
  exit 1
}
[ -d $R/Dataset904_ImageCAS_S4 ] || tar --no-same-owner -xf /workspace/upload/split34_resencl_ckpts.tar -C $R || {
  log "LOI tar split34"
  exit 1
}
summ() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print({k: round(v['mean'],4) for k,v in r.items() if isinstance(v,dict)})" $1; }
TR=nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres
for k in 2 3 4; do
  DS=Dataset90${k}_ImageCAS_S$k
  [ -f $W/rerun_split${k}_result.json ] && continue
  log "ckpt S$k sha256 $(sha256sum $R/$DS/$TR/fold_0/checkpoint_final.pth | cut -c1-64)"
  python3 $C/src/coronary_tree/data/prepare.py --manifest $C/configs/imagecas_v2_split_manifest.csv --cases-dir /workspace/cases --split $k --dataset-id 90$k --nnunet-raw $W/nnunet_raw --nnunet-preprocessed $W/nnunet_preprocessed --work $W --stage test --workers 32 >> $LOG 2>&1 || {
    log "LOI prepare S$k"
    exit 1
  }
  nnUNetv2_predict -i $W/nnunet_raw/$DS/imagesTs -o $W/split$k/test_pred -d 90$k -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans -chk checkpoint_final.pth -step_size 0.25 -npp 4 -nps 4 > /workspace/logs/p3_pred_s$k.log 2>&1 || {
    log "LOI predict S$k"
    exit 1
  }
  python3 $C/scripts/score_segmentation.py $W/split$k/test_pred $W/test_labels_s$k $W/rerun_split${k}_result.json --workers 32 >> $LOG 2>&1 || {
    log "LOI score S$k"
    exit 1
  }
  log "S$k-DONE: $(summ $W/rerun_split${k}_result.json)"
done
S1=$W/testaccess3/test/resenc_l__checkpoint_final__s0.25
if [ -d $S1 ] && [ ! -f $W/rerun_split1_result.json ]; then
  mkdir -p $W/split1/test_pred
  for f in $S1/*.nii.gz; do ln -f $f $W/split1/test_pred/$(basename $f); done
  python3 $C/scripts/score_segmentation.py $W/split1/test_pred $W/test_labels_s1 $W/rerun_split1_result.json --workers 32 >> $LOG 2>&1 && log "S1-DONE: $(summ $W/rerun_split1_result.json)" || log "LOI score S1"
fi
python3 - << 'PYSUM' >> $LOG 2>&1
import csv, statistics
W = "/workspace/coroseg-work"
means, allrows = {}, []
for k in (1, 2, 3, 4):
    try:
        rows = list(csv.DictReader(open(f"{W}/rerun_split{k}_result.per_case.csv", encoding="utf-8-sig")))
    except FileNotFoundError:
        continue
    d = [float(r["dice"]) for r in rows]
    means[k] = statistics.mean(d)
    allrows += d
print("FOURSPLIT single ResEnc-L:", {k: round(v, 4) for k, v in means.items()}, "mean of split means", round(statistics.mean(means.values()), 4) if means else None, "patient-wise", round(statistics.mean(allrows), 4) if allrows else None, "n", len(allrows))
PYSUM
log "P3-DONE"
