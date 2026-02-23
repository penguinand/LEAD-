#!/bin/bash
# Office-31 OPDA: Source model training (all 3 domains)
set -e
cd "$(dirname "$0")/.."

for s_idx in 0 1 2; do
    echo "=== Training source model: Office domain $s_idx OPDA ==="
    python3 train_source.py --dataset Office --s_idx $s_idx \
        --target_label_type OPDA --epochs 50 --lr 0.01 --batch_size 64 --num_workers 0
done
