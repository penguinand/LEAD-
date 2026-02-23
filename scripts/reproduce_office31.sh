#!/bin/bash
# Master reproduction script for LEAD on Office-31
# Runs all source training + target adaptation for OPDA, OSDA, PDA
set -e
cd "$(dirname "$0")"

NW=0  # num_workers=0 for macOS compatibility

echo "============================================"
echo "  LEAD Reproduction: Office-31 Full Pipeline"
echo "============================================"

###############################################
# Phase 1: Source Model Training
###############################################

echo ""
echo ">>> Phase 1: Training ALL source models <<<"
echo ""

# OPDA sources
echo "[1/9] Source 0 OPDA..."
python3 train_source.py --dataset Office --s_idx 0 --target_label_type OPDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[2/9] Source 1 OPDA..."
python3 train_source.py --dataset Office --s_idx 1 --target_label_type OPDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[3/9] Source 2 OPDA..."
python3 train_source.py --dataset Office --s_idx 2 --target_label_type OPDA --epochs 50 --lr 0.01 --num_workers $NW

# OSDA sources
echo "[4/9] Source 0 OSDA..."
python3 train_source.py --dataset Office --s_idx 0 --target_label_type OSDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[5/9] Source 1 OSDA..."
python3 train_source.py --dataset Office --s_idx 1 --target_label_type OSDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[6/9] Source 2 OSDA..."
python3 train_source.py --dataset Office --s_idx 2 --target_label_type OSDA --epochs 50 --lr 0.01 --num_workers $NW

# PDA sources
echo "[7/9] Source 0 PDA..."
python3 train_source.py --dataset Office --s_idx 0 --target_label_type PDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[8/9] Source 1 PDA..."
python3 train_source.py --dataset Office --s_idx 1 --target_label_type PDA --epochs 50 --lr 0.01 --num_workers $NW
echo "[9/9] Source 2 PDA..."
python3 train_source.py --dataset Office --s_idx 2 --target_label_type PDA --epochs 50 --lr 0.01 --num_workers $NW

###############################################
# Phase 2: Target Adaptation - OPDA
###############################################

echo ""
echo ">>> Phase 2: OPDA Target Adaptation <<<"
echo ""

LAM=0.30
echo "[OPDA 1/6] A→D"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW
echo "[OPDA 2/6] A→W"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW
echo "[OPDA 3/6] D→A"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW
echo "[OPDA 4/6] D→W"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW
echo "[OPDA 5/6] W→A"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW
echo "[OPDA 6/6] W→D"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type OPDA --num_workers $NW

###############################################
# Phase 3: Target Adaptation - OSDA
###############################################

echo ""
echo ">>> Phase 3: OSDA Target Adaptation <<<"
echo ""

LAM=0.30
echo "[OSDA 1/6] A→D"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW
echo "[OSDA 2/6] A→W"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW
echo "[OSDA 3/6] D→A"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW
echo "[OSDA 4/6] D→W"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW
echo "[OSDA 5/6] W→A"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW
echo "[OSDA 6/6] W→D"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type OSDA --num_workers $NW

###############################################
# Phase 4: Target Adaptation - PDA
###############################################

echo ""
echo ">>> Phase 4: PDA Target Adaptation <<<"
echo ""

LAM=0.30
echo "[PDA 1/6] A→D"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW
echo "[PDA 2/6] A→W"
python3 train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW
echo "[PDA 3/6] D→A"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW
echo "[PDA 4/6] D→W"
python3 train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW
echo "[PDA 5/6] W→A"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW
echo "[PDA 6/6] W→D"
python3 train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --lam_psd $LAM --target_label_type PDA --num_workers $NW

echo ""
echo "============================================"
echo "  ALL TRAINING COMPLETE!"
echo "============================================"
