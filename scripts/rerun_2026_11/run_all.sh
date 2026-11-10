#!/bin/bash
# Watcher: runs the three rerun phases as soon as their inputs are on the pod. Markers in /workspace/rerun.log.
set -u
LOG=/workspace/rerun.log
log() { echo "$(date +%F_%T) ALL $*" >> $LOG; }
log "run_all start"
until grep -q EXTRACT-DONE /workspace/bootstrap.log 2> /dev/null; do sleep 60; done
log "ImageCAS cases ready"
until [ -f /workspace/upload/.done_xzip ] && [ -f /workspace/upload/.done_lumen ]; do sleep 60; done
if bash /workspace/rerun/phase1_lumen.sh; then log "phase1 ok"; else log "phase1 FAILED"; fi
until [ -f /workspace/upload/.done_ft500 ] && [ -f /workspace/upload/.done_resencL ]; do sleep 60; done
if bash /workspace/rerun/phase2_ensemble_s1.sh; then log "phase2 ok"; else log "phase2 FAILED"; fi
until [ -f /workspace/upload/.done_split2 ] && [ -f /workspace/upload/.done_split34 ]; do sleep 60; done
if bash /workspace/rerun/phase3_splits.sh; then log "phase3 ok"; else log "phase3 FAILED"; fi
log "run_all end"
