#!/bin/bash
# Master reproduction script for LEAD Jittor version
# Runs Office-31 (OPDA/OSDA/PDA) + OfficeHome (OPDA) target adaptation
set -e
cd "$(dirname "$0")/.."

EPOCHS=15

echo "============================================"
echo "  LEAD Jittor Reproduction - Started"
echo "  Epochs per pair: $EPOCHS"
echo "============================================"
date

# Office-31 OPDA
echo ""
echo ">>> Phase 1: Office-31 OPDA (6 pairs)"
for s_idx in 0 1 2; do
    for t_idx in 0 1 2; do
        if [ $s_idx -ne $t_idx ]; then
            echo "--- Office OPDA: s=$s_idx -> t=$t_idx ---"
            python3 train_target.py --dataset Office --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OPDA --epochs $EPOCHS --lr 0.001 --lam_psd 0.30 \
                --batch_size 64 --num_workers 0
        fi
    done
done

# Office-31 OSDA
echo ""
echo ">>> Phase 2: Office-31 OSDA (6 pairs)"
for s_idx in 0 1 2; do
    for t_idx in 0 1 2; do
        if [ $s_idx -ne $t_idx ]; then
            echo "--- Office OSDA: s=$s_idx -> t=$t_idx ---"
            python3 train_target.py --dataset Office --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OSDA --epochs $EPOCHS --lr 0.001 --lam_psd 0.30 \
                --batch_size 64 --num_workers 0
        fi
    done
done

# Office-31 PDA
echo ""
echo ">>> Phase 3: Office-31 PDA (6 pairs)"
for s_idx in 0 1 2; do
    for t_idx in 0 1 2; do
        if [ $s_idx -ne $t_idx ]; then
            echo "--- Office PDA: s=$s_idx -> t=$t_idx ---"
            python3 train_target.py --dataset Office --s_idx $s_idx --t_idx $t_idx \
                --target_label_type PDA --epochs $EPOCHS --lr 0.001 --lam_psd 0.30 \
                --batch_size 64 --num_workers 0
        fi
    done
done

# OfficeHome OPDA
echo ""
echo ">>> Phase 4: OfficeHome OPDA (12 pairs)"
for s_idx in 0 1 2 3; do
    for t_idx in 0 1 2 3; do
        if [ $s_idx -ne $t_idx ]; then
            echo "--- OfficeHome OPDA: s=$s_idx -> t=$t_idx ---"
            python3 train_target.py --dataset OfficeHome --s_idx $s_idx --t_idx $t_idx \
                --target_label_type OPDA --epochs $EPOCHS --lr 0.001 --lam_psd 2.00 \
                --batch_size 64 --num_workers 0
        fi
    done
done

echo ""
echo "============================================"
echo "  LEAD Jittor Reproduction - Completed"
echo "============================================"
date
