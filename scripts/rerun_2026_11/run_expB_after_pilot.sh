#!/bin/bash
# Watcher (03/11/2026, user decision "chờ pilot xong rồi chạy"): start experiment B as soon as the pilot has written
# PILOT-DONE (or LOI) in /workspace/pilot.log, i.e. when the GPU is free. Logs to /workspace/expB.log.
LOG=/workspace/expB.log
echo "$(date +%F_%T) B WATCH start: waiting for PILOT-DONE" >> $LOG
while ! grep -q -E "PILOT (PILOT-DONE|LOI)" /workspace/pilot.log; do sleep 300; done
echo "$(date +%F_%T) B WATCH pilot marker: $(grep -E "PILOT (PILOT-DONE|LOI)" /workspace/pilot.log | tail -1 | cut -c1-120)" >> $LOG
sleep 60
exec bash /workspace/rerun/expB_skelrec_s1.sh
