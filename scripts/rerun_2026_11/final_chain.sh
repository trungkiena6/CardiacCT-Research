#!/bin/bash
# Final chain (PREREG_final_split1.md, 03/11/2026, user: "tiếp tục & hoàn thiện"):
#   wait for experiment B to finish (B itself waits for the pilot) -> val-stage selection over the pool -> gate
#   -> [seed replicates 2, 3 while the gate fails] -> sixth Split-1 test access only if the gate passes.
# Markers in /workspace/final6.log (prefix F). Re-running resumes: finished members/ensembles/choices are reused.
set -u
C=/workspace/coroseg
W=/workspace/coroseg-work
LOG=/workspace/final6.log
log() { echo "$(date +%F_%T) F $*" >> $LOG; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=$W/nnunet_raw nnUNet_preprocessed=$W/nnunet_preprocessed nnUNet_results=$W/nnunet_results nnUNet_compile=false
export PYTHONPATH=$C/src COROSEG_WORK=$W
mkdir -p /workspace/logs
cd $C/scripts || exit 1
stage() { python3 /workspace/rerun/final6_pool.py $1 >> /workspace/logs/final6_$1.log 2>&1; }
gate() { python3 -c "import json,glob,os; f=sorted(glob.glob('$W/final6/val_choice_pool*.json'), key=os.path.getmtime); print(json.load(open(f[-1]))['gate'] if f else 'NONE')"; }

if [ "${CHAIN_SKIP_WAIT:-0}" != "1" ]; then
  log "CHAIN start: waiting for B-DONE/LOI in /workspace/expB.log (B waits for PILOT-DONE)"
  while ! grep -q -E " B (B-DONE|LOI)" /workspace/expB.log; do sleep 300; done
  log "CHAIN B marker: $(grep -E ' B (B-DONE|LOI)' /workspace/expB.log | tail -1 | cut -c1-140)"
  grep -q -E "PILOT (PILOT-DONE|LOI)" /workspace/pilot.log || {
    log "LOI pilot has no final marker"
    exit 1
  }
  sleep 60
fi
while pgrep -f "/nnUNetv2_(train|predict) " > /dev/null; do
  log "waiting: GPU still busy"
  sleep 300
done

stage val || {
  log "LOI val stage"
  exit 1
}
if [ "$(gate)" != "PASS" ]; then
  for k in 2 3; do
    log "GATE FAIL -> experiment C seed$k"
    bash /workspace/rerun/expC_seed.sh $k || {
      log "LOI seed$k"
      exit 1
    }
    stage val || {
      log "LOI val stage after seed$k"
      exit 1
    }
    [ "$(gate)" = "PASS" ] && break
  done
fi
if [ "$(gate)" = "PASS" ]; then
  stage selfcheck || {
    log "LOI selfcheck"
    exit 1
  }
  stage test || {
    log "LOI test stage"
    exit 1
  }
else
  log "GATE FAIL after the full pool: the test set is NOT touched; user decision needed"
fi
log "CHAIN-DONE"
