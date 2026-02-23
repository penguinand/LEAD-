#!/bin/bash
# Office-31 OSDA: Target adaptation (all 6 domain pairs)
set -e
cd "$(dirname "$0")/.."

for s_idx in 0 1 2; do
    for t_idx in 0 1 2; do
        if [ $s_idx -ne $t_idx ]; then
            echo "=== Target adaptation: Office s=$s_idx -> t=$t_idx OSDA ==="
            python3 train_target.py --dataset Office --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OSDA --epochs 50 --lr 0.001 --lam_psd 0.30 \
                --batch_size 64 --num_workers 0
        fi
    done
done
