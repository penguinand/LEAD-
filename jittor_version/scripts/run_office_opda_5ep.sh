#!/bin/bash
# Office-31 OPDA: 5 epochs, all 6 domain pairs
set -e
cd "$(dirname "$0")/.."

EPOCHS=5
echo "=== Office-31 OPDA (6 pairs, ${EPOCHS} epochs) ==="
date

for s_idx in 0 1 2; do
    for t_idx in 0 1 2; do
        if [ $s_idx -ne $t_idx ]; then
            echo ""
            echo "--- Office OPDA: s=$s_idx -> t=$t_idx ---"
            python3 train_target.py --dataset Office --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OPDA --epochs $EPOCHS --lr 0.001 --lam_psd 0.30 \
                --batch_size 64 --num_workers 0
        fi
    done
done

echo ""
echo "=== All Office-31 OPDA pairs completed ==="
date
