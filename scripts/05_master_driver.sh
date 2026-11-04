#!/usr/bin/env bash
set -euo pipefail
ROOT="${BRIM_DATA_DIR:-./data}"
assert_prefix="${BRIM_DATA_DIR:-./data}"
if [[ "$ROOT" != "$assert_prefix" ]]; then
  echo "FATAL: ROOT mismatch" >&2
  exit 1
fi
PY="python3"

echo "[$(date +%H:%M:%S)] === STEP 03: CODON POSITIVE SELECTION (TIER 1 & TIER 2) ==="
"$PY" "$ROOT/00_scripts/03_run_codon_selection.py" 2>&1 | tee "$ROOT/logs/03_run_codon_selection.log"

echo "[$(date +%H:%M:%S)] === STEP 03b: RECOMBINATION / PARTITIONED TREE SENSITIVITY ==="
"$PY" "$ROOT/00_scripts/03b_recombination_gard_sensitivity.py" 2>&1 | tee "$ROOT/logs/03b_recombination_gard_sensitivity.log"

echo "[$(date +%H:%M:%S)] === STEP 04b: STAGE 0 KILL INTEGRATION (K1 + K2 + K3) ==="
"$PY" "$ROOT/00_scripts/04b_stage0_kill_integration.py" 2>&1 | tee "$ROOT/logs/04b_stage0_kill_integration.log"

echo "[$(date +%H:%M:%S)] === ALL_PIPELINE_STEPS_COMPLETE ==="
