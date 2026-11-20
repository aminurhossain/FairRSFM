#!/usr/bin/env bash
# ==============================================================================
# Robust Sequential Queue for TerraFM on Stratified 14-Biome CloudSEN12
# Methods: ERM -> Group DRO -> DBR -> BOLP
# ==============================================================================

set +e  # Do NOT exit if a single job fails; continue queue and log clearly

export PYTHONPATH="/data1/aminur/om/prithvi/FairRSFM_repo/src:${PYTHONPATH}"
export PROJ_DATA="/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
export PROJ_LIB="/home/aminur/anaconda3/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/data1/aminur/om/prithvi/FairRSFM_repo/Final_cloudsen12/terrafm"
mkdir -p "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/cloudsen12_terrafm_queue.log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "=============================================================================="
echo " Starting TerraFM CloudSEN12 14-Biome Queue at $(date)"
echo " Log file: ${QUEUE_LOG}"
echo " GPU     : 0 (NVIDIA RTX A6000 48GB)"
echo " Epochs  : 50 (NO early stopping, Val & Test eval saved for every epoch)"
echo "=============================================================================="

METHODS=("erm" "groupdro" "dbr" "bolp")

for METHOD in "${METHODS[@]}"; do
    METHOD_OUT="${LOG_DIR}/${METHOD}"
    mkdir -p "${METHOD_OUT}"

    if [[ -f "${METHOD_OUT}/test_selected.json" && -f "${METHOD_OUT}/epoch_eval_history.json" && -f "${METHOD_OUT}/conf_mat_14x4x4.npy" ]]; then
        echo ""
        echo ">>> [$(date)] METHOD=${METHOD^^} on CloudSEN12 is ALREADY COMPLETED! Skipping."
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo ">>> [$(date)] Starting TerraFM | Method: ${METHOD^^} on CloudSEN12"
    echo "------------------------------------------------------------------------------"

    python3 "${SCRIPT_DIR}/run_cloudsen12.py" \
        --model terrafm \
        --method "${METHOD}" \
        --mode train \
        --batch-size 16 \
        --epochs 50 \
        --lr 1e-3 \
        --workers 4 \
        --seed 42 \
        --output-dir "${METHOD_OUT}" \
        --gpu 0

    STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
        echo ">>> [$(date)] SUCCESS: ${METHOD^^} completed cleanly (exit code 0)."
        if [ -f "${METHOD_OUT}/test_selected.json" ]; then
            echo "    ✓ Verified test_selected.json artifact generated."
        else
            echo "    ⚠ Warning: test_selected.json not found in ${METHOD_OUT}"
        fi
    else
        echo ">>> [$(date)] ERROR: ${METHOD^^} exited with code ${STATUS}. Continuing to next method in queue..."
    fi
    echo "------------------------------------------------------------------------------"
done

echo ""
echo "=============================================================================="
echo " TerraFM CloudSEN12 Queue Finished at $(date)"
echo "=============================================================================="
