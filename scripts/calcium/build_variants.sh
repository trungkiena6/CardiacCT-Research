#!/bin/bash
# Sau khi co mask tim: dung Dataset911 (crop) va 912 (crop + kenh ung vien) tu Dataset910, crop tap val/test, roi preprocess 911/912 (CPU).
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/workspace/cacseg-work
LOG=/workspace/cacseg/build_variants.log
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results
until grep -q "HEART-DONE" $W/heart/_progress.log 2> /dev/null; do sleep 120; done
mkdir -p $W/heart_flat
for d in $W/heart/*/; do
  s=$(basename $d)
  [ -f $d/heart.nii.gz ] && ln -sf $d/heart.nii.gz $W/heart_flat/$s.nii.gz
done
echo "$(date +%F_%T) heart_flat: $(ls $W/heart_flat | wc -l)" >> $LOG
cd /workspace/cacseg && python3 make_cacseg_variants.py --src $W/nnunet_raw/Dataset910_COCA_CAC_HO --heart-dir $W/heart_flat --coca-root /workspace/coca/cocacoronarycalciumandchestcts-2 --out-root $W/nnunet_raw >> $LOG 2>&1
cd $W/nnunet_raw && rm -rf Dataset906_COCA_CAC_BRANCH Dataset911_COCA_CAC_HO_CROP Dataset912_COCA_CAC_HO_CAND && mv Dataset904_COCA_CAC_CROP Dataset911_COCA_CAC_HO_CROP && mv Dataset905_COCA_CAC_CAND Dataset912_COCA_CAC_HO_CAND
sed -i "s/Dataset904_COCA_CAC_CROP/Dataset911_COCA_CAC_HO_CROP/" Dataset911_COCA_CAC_HO_CROP/dataset.json
sed -i "s/Dataset905_COCA_CAC_CAND/Dataset912_COCA_CAC_HO_CAND/" Dataset912_COCA_CAC_HO_CAND/dataset.json
echo "$(date +%F_%T) 911 imagesTr: $(ls Dataset911_COCA_CAC_HO_CROP/imagesTr | wc -l) | 912 imagesTr: $(ls Dataset912_COCA_CAC_HO_CAND/imagesTr | wc -l)" >> $LOG
cd /workspace/cacseg && python3 crop_eval_sets.py >> $LOG 2>&1
for d in 911:Dataset911_COCA_CAC_HO_CROP 912:Dataset912_COCA_CAC_HO_CAND; do
  id=${d%%:*}
  name=${d#*:}
  mkdir -p $W/nnunet_preprocessed/$name && cp $W/splits910_backup.json $W/nnunet_preprocessed/$name/splits_final.json
  nnUNetv2_plan_and_preprocess -d $id -c 3d_fullres --verify_dataset_integrity -np 16 > /workspace/cacseg/preprocess$id.log 2>&1 && echo "PREPROC$id-DONE" >> $LOG || echo "PREPROC$id-LOI" >> $LOG
  cp $W/splits910_backup.json $W/nnunet_preprocessed/$name/splits_final.json
done
echo "$(date +%F_%T) BUILD-VARIANTS-DONE" >> $LOG
