#!/bin/bash
# OfficeHome OPDA: Target adaptation (all 12 domain pairs)
set -e
cd "$(dirname "$0")/.."

for s_idx in 0 1 2 3; do
    for t_idx in 0 1 2 3; do
        if [ $s_idx -ne $t_idx ]; then
            echo "=== Target adaptation: OfficeHome s=$s_idx -> t=$t_idx OPDA ==="
            python3 train_target.py --dataset OfficeHome --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OPDA --epochs 50 --lr 0.001 --lam_psd 2.00 \
                --batch_size 64 --num_workers 0
        fi
    done
done
