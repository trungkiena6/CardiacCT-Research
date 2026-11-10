#!/bin/bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=/workspace/cacseg-work/nnunet_raw nnUNet_preprocessed=/workspace/cacseg-work/nnunet_preprocessed nnUNet_results=/workspace/cacseg-work/nnunet_results
cp /workspace/cacseg-work/nnunet_preprocessed/Dataset910_COCA_CAC_HO/splits_final.json /workspace/cacseg-work/splits910_backup.json
nnUNetv2_plan_and_preprocess -d 910 -c 3d_fullres --verify_dataset_integrity -np 16 > /workspace/cacseg/preprocess910.log 2>&1 && echo "PREPROC910-DONE" >> /workspace/cacseg/preprocess910.log || echo "PREPROC910-LOI" >> /workspace/cacseg/preprocess910.log
# nnU-Net co the ghi de splits_final khi train; khoi phuc ban 1 fold cua minh
cp /workspace/cacseg-work/splits910_backup.json /workspace/cacseg-work/nnunet_preprocessed/Dataset910_COCA_CAC_HO/splits_final.json
python3 -c "import json; s=json.load(open(\"/workspace/cacseg-work/nnunet_preprocessed/Dataset910_COCA_CAC_HO/splits_final.json\")); print(\"splits:\", len(s), \"fold(s); train\", len(s[0][\"train\"]), \"val\", len(s[0][\"val\"]))" >> /workspace/cacseg/preprocess910.log
