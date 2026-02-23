#!/usr/bin/env python3
"""Collect and format LEAD Jittor reproduction results."""
import os
import re
import sys

def extract_best_scores(log_path):
    """Extract final Best H-Score, KnownAcc, UnknownAcc from log file."""
    best_line = None
    with open(log_path, 'r') as f:
        for line in f:
            if 'Best   :' in line:
                best_line = line.strip()
    if best_line is None:
        return None
    m = re.search(r'H-Score:([\d.]+), KnownAcc:([\d.]+), UnknownAcc:([\d.]+)', best_line)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Also check PyTorch results for comparison
    pt_base = os.path.join(base_dir, '..', 'checkpoints')
    jt_base = os.path.join(base_dir, 'checkpoints')
    
    datasets_scenarios = [
        ('Office', 'OPDA', [0,1,2], 0.3),
        ('Office', 'OSDA', [0,1,2], 0.3),
        ('Office', 'PDA', [0,1,2], 0.3),
        ('OfficeHome', 'OPDA', [0,1,2,3], 2.0),
    ]
    
    domain_names = {
        'Office': {0: 'A', 1: 'D', 2: 'W'},
        'OfficeHome': {0: 'Ar', 1: 'Cl', 2: 'Pr', 3: 'Rw'},
    }
    
    for dataset, scenario, domains, lam_psd in datasets_scenarios:
        print(f"\n{'='*60}")
        print(f"  {dataset} - {scenario}")
        print(f"{'='*60}")
        print(f"{'Pair':<10} {'Jittor H':>10} {'Jittor K':>10} {'Jittor U':>10} {'PyTorch H':>10}")
        print(f"{'-'*50}")
        
        jt_scores = []
        pt_scores = []
        
        for s in domains:
            for t in domains:
                if s == t:
                    continue
                sn = domain_names[dataset][s]
                tn = domain_names[dataset][t]
                pair_name = f"{sn}->{tn}"
                
                # Jittor result
                jt_log = os.path.join(jt_base, dataset, f"s_{s}_t_{t}", scenario,
                                       f"smooth_psd_{lam_psd}", "log_target_training.txt")
                jt_result = extract_best_scores(jt_log) if os.path.exists(jt_log) else None
                
                # PyTorch result
                pt_log = os.path.join(pt_base, dataset, f"s_{s}_t_{t}", scenario,
                                       f"smooth_psd_{lam_psd}", "log_target_training.txt")
                pt_result = extract_best_scores(pt_log) if os.path.exists(pt_log) else None
                
                jt_h = f"{jt_result[0]:.3f}" if jt_result else "---"
                jt_k = f"{jt_result[1]:.3f}" if jt_result else "---"
                jt_u = f"{jt_result[2]:.3f}" if jt_result else "---"
                pt_h = f"{pt_result[0]:.3f}" if pt_result else "---"
                
                print(f"{pair_name:<10} {jt_h:>10} {jt_k:>10} {jt_u:>10} {pt_h:>10}")
                
                if jt_result:
                    jt_scores.append(jt_result[0])
                if pt_result:
                    pt_scores.append(pt_result[0])
        
        if jt_scores:
            jt_avg = sum(jt_scores) / len(jt_scores)
            pt_avg = sum(pt_scores) / len(pt_scores) if pt_scores else 0
            print(f"{'-'*50}")
            print(f"{'AVG':<10} {jt_avg:>10.3f} {'':>10} {'':>10} {pt_avg:>10.3f}")

if __name__ == "__main__":
    main()
