#!/bin/bash
# Out-of-sample CoroSeg predictions for the 800 ImageCAS-X cases with the official 4-split ResEnc-L models: each case is
# predicted by the model of the split whose TEST set contains it (the four test sets partition the 1000 ImageCAS cases).
# Waits for the checkpoint upload (ckpts_*.tar) and for DATA-READY (extracted cases). step 0.5, no TTA, thr 0.5, no component removal.
# Markers in /workspace/oos.log: "predict split k", OOS-DONE, LOI on failure.
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_PYTHON="${CORONARY_PYTHON:-python3}"
W="${COROSEG_OOS_WORK:-/workspace/oos}"
LOG=/workspace/oos.log
mkdir -p "$W"
log() { echo "$(date +%F_%T) $*" >> $LOG; }
require_fresh_directory() {
  local directory=$1 entries
  if [[ -e "$directory" || -L "$directory" ]]; then
    [[ -d "$directory" ]] || {
      echo "Output is not a directory: $directory" >&2
      exit 1
    }
    entries=$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit) || {
      echo "Cannot inspect output: $directory" >&2
      exit 1
    }
    [[ -z "$entries" ]] || {
      echo "Existing output preserved; select a fresh COROSEG_OOS_WORK: $directory" >&2
      exit 1
    }
  fi
}
for split in 1 2 3 4; do require_fresh_directory "$W/pred_$split"; done
require_fresh_directory "$W/pred"
export COROSEG_OOS_WORK="$W"
export COROSEG_IMAGECAS_MANIFEST="${COROSEG_IMAGECAS_MANIFEST:-$TRAINING_SCRIPT_DIR/../../configs/imagecas_v2_split_manifest.csv}"
export OMP_NUM_THREADS=1 nnUNet_raw=$W/raw nnUNet_preprocessed=$W/pre nnUNet_results=$W/results nnUNet_compile=false
until [ -f /workspace/ckpts_split34_resencl_ckpts.tar ] && [ "$(stat -c %s /workspace/ckpts_split34_resencl_ckpts.tar)" -ge 2264791040 ]; do sleep 60; done
until grep -q DATA-READY /workspace/xbench.log 2> /dev/null; do sleep 120; done
if [ ! -d $W/results/Dataset904_ImageCAS_S4 ]; then
  mkdir -p $W/results/Dataset901_ImageCAS_S1
  tar --no-same-owner -xf /workspace/ckpts_resencL_ckpts.tar -C $W/results/Dataset901_ImageCAS_S1 nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres || {
    log "LOI tar S1"
    exit 1
  }
  tar --no-same-owner -xf /workspace/ckpts_split2_resencl_ckpt.tar -C $W/results || {
    log "LOI tar S2"
    exit 1
  }
  tar --no-same-owner -xf /workspace/ckpts_split34_resencl_ckpts.tar -C $W/results || {
    log "LOI tar S34"
    exit 1
  }
  log "checkpoints: $(ls $W/results)"
fi
"$WORKFLOW_PYTHON" - << 'PY_OOS_INPUTS' >> $LOG 2>&1 || {
import csv, hashlib, json, os
from pathlib import Path
work = Path(os.environ['COROSEG_OOS_WORK'])
X = Path(os.environ.get('COROSEG_IMAGECASX_FILELIST', '/workspace/xdata/ImageCAS-X_dataset/filelist'))
groups = [(X / f'{role}.txt').read_text().split() for role in ('train', 'val', 'test')]
if tuple(map(len, groups)) != (560, 80, 160):
    raise ValueError('ImageCAS-X requires the official 560/80/160 split')
ids = [case for group in groups for case in group]
if len(set(ids)) != 800:
    raise ValueError('ImageCAS-X case IDs must be distinct across all partitions')
with Path(os.environ['COROSEG_IMAGECAS_MANIFEST']).open(encoding='utf-8-sig', newline='') as handle:
    rows = {r['case_id']: r for r in csv.DictReader(handle)}
scope = {str(k): [] for k in (1, 2, 3, 4)}
cases = Path(os.environ.get('COROSEG_IMAGECAS_CASES', '/workspace/cases'))
def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()
for i in ids:
    ks = [k for k in (1, 2, 3, 4) if rows[i][f'split_{k}'] == 'test']
    if len(ks) != 1:
        raise ValueError(f'{i}: expected exactly one held-out ImageCAS model')
    scope[str(ks[0])].append(i)
    b = (int(i) - 1) // 200 * 200
    src = cases / f'{b + 1}-{b + 200}/{i}.img.nii.gz'
    dst = work / f'in_{ks[0]}/{i}_0000.nii.gz'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not os.path.samefile(src, dst) and digest(src) != digest(dst):
            raise FileExistsError(f'{dst}: stale input; select a fresh work root')
    else:
        os.link(src, dst)
for split, expected in scope.items():
    actual = {path.name[:-12] for path in (work / f'in_{split}').glob('*_0000.nii.gz')}
    if actual != set(expected) or not expected:
        raise ValueError(f'Split {split}: input case set differs from the official scope')
scope_path = work / 'oos_case_ids.json'
if scope_path.exists():
    if json.loads(scope_path.read_text(encoding='utf-8')) != scope:
        raise FileExistsError('Existing OOS scope differs; select a fresh work root')
else:
    with scope_path.open('x', encoding='utf-8') as handle:
        json.dump(scope, handle, indent=2)
print('inputs', {split: len(ids) for split, ids in scope.items()})
PY_OOS_INPUTS
  log "LOI inputs"
  exit 1
}
for k in 1 2 3 4; do
  DS=$(ls $W/results | grep "_S$k$")
  n_in=$(ls $W/in_$k | wc -l)
  log "predict split $k ($DS): $n_in ca"
  nnUNetv2_predict -i $W/in_$k -o $W/pred_$k -d $DS -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_NoMirroring -p nnUNetResEncUNetL40GPlans -chk checkpoint_final.pth -step_size 0.5 --disable_tta -npp 3 -nps 3 > /workspace/oos_pred_$k.log 2>&1 || {
    log "LOI predict $k"
    exit 1
  }
done
"$WORKFLOW_PYTHON" - << 'PY_OOS_MERGE' >> $LOG 2>&1 || {
import json, os
from pathlib import Path
work = Path(os.environ['COROSEG_OOS_WORK'])
scope = json.loads((work / 'oos_case_ids.json').read_text(encoding='utf-8'))
if set(scope) != {'1', '2', '3', '4'}:
    raise ValueError('OOS scope must identify all four splits')
all_ids = [case for ids in scope.values() for case in ids]
if not all_ids or len(all_ids) != len(set(all_ids)) or any(not ids for ids in scope.values()):
    raise ValueError('OOS case sets must be nonempty and disjoint')
for split, expected in scope.items():
    actual = {path.name[:-7] for path in (work / f'pred_{split}').glob('*.nii.gz')}
    if actual != set(expected):
        raise ValueError(f'Split {split}: prediction cases differ from the required scope')
merged = work / 'pred'
if merged.exists() and any(merged.iterdir()):
    raise FileExistsError('Merged output is nonempty; existing masks were preserved')
merged.mkdir(exist_ok=True)
for split, ids in scope.items():
    for case in ids:
        os.link(work / f'pred_{split}/{case}.nii.gz', merged / f'{case}.nii.gz')
if {path.name[:-7] for path in merged.glob('*.nii.gz')} != set(all_ids):
    raise ValueError('Merged prediction scope is incomplete')
PY_OOS_MERGE
  log "LOI merge predictions"
  exit 1
}
log "OOS-DONE: $(ls $W/pred | wc -l) predictions"
