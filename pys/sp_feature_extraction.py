#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse

# =========================
# 氨基酸属性定义
# =========================

HYDROPHOBIC_AA = set("AILMFWVY")
SMALL_AA = set("ASG")
POSITIVE = set("KR")
NEGATIVE = set("DE")

GRAVY_SCALE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# =========================
# FASTA 读取
# =========================

def read_fasta(fasta_file):
    seq_dict = {}
    with open(fasta_file) as f:
        current_id = None
        seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    seq_dict[current_id] = "".join(seq)
                current_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.upper())

        if current_id:
            seq_dict[current_id] = "".join(seq)

    return seq_dict


# =========================
# 工具函数
# =========================

def find_h_region(seq):
    """找最长连续疏水区"""
    max_start, max_end = -1, -1
    current_start = None

    for i, aa in enumerate(seq):
        if aa in HYDROPHOBIC_AA:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                if (i - current_start) > (max_end - max_start):
                    max_start, max_end = current_start, i
                current_start = None

    # 收尾
    if current_start is not None:
        if (len(seq) - current_start) > (max_end - max_start):
            max_start, max_end = current_start, len(seq)

    return max_start, max_end


def calc_charge(seq):
    pos = sum(aa in POSITIVE for aa in seq)
    neg = sum(aa in NEGATIVE for aa in seq)
    return pos - neg


def calc_gravy(seq):
    if len(seq) == 0:
        return np.nan
    return np.mean([GRAVY_SCALE.get(aa, 0) for aa in seq])


def small_aa_ratio(seq):
    if len(seq) == 0:
        return np.nan
    return sum(aa in SMALL_AA for aa in seq) / len(seq)


# =========================
# 核心：SP 切分
# =========================

def split_sp_regions(row, seq_dict):
    if row['has_SP'] == 0:
        return pd.Series({
            'N_len': np.nan,
            'H_len': np.nan,
            'C_len': np.nan,
            'N_charge': np.nan,
            'H_gravy': np.nan,
            'C_small_ratio': np.nan
        })

    protein_id = row['protein_id']
    seq = seq_dict.get(protein_id, "")

    if not seq:
        return pd.Series({
            'N_len': np.nan,
            'H_len': np.nan,
            'C_len': np.nan,
            'N_charge': np.nan,
            'H_gravy': np.nan,
            'C_small_ratio': np.nan
        })

    # 安全检查
    start = int(row['SP_start'])
    end = int(row['SP_end'])

    if start <= 0 or end <= 0 or end > len(seq):
        return pd.Series([np.nan]*6,
                         index=['N_len','H_len','C_len','N_charge','H_gravy','C_small_ratio'])

    # 1-based → python index
    sp_seq = seq[start-1:end]

    if len(sp_seq) == 0:
        return pd.Series([np.nan]*6,
                         index=['N_len','H_len','C_len','N_charge','H_gravy','C_small_ratio'])

    # 找 H-region
    h_start, h_end = find_h_region(sp_seq)

    if h_start == -1:
        return pd.Series([np.nan]*6,
                         index=['N_len','H_len','C_len','N_charge','H_gravy','C_small_ratio'])

    # 切分
    N_region = sp_seq[:h_start]
    H_region = sp_seq[h_start:h_end]
    C_region = sp_seq[h_end:]

    return pd.Series({
        'N_len': len(N_region),
        'H_len': len(H_region),
        'C_len': len(C_region),
        'N_charge': calc_charge(N_region),
        'H_gravy': calc_gravy(H_region),
        'C_small_ratio': small_aa_ratio(C_region)
    })


# =========================
# 主程序
# =========================

def main():
    parser = argparse.ArgumentParser(description="SP region feature extraction")
    parser.add_argument("--pred", required=True, help="prediction TSV")
    parser.add_argument("--fasta", required=True, help="protein FASTA")
    parser.add_argument("--out", required=True, help="output TSV")

    args = parser.parse_args()

    print(">>> Reading prediction file...")
    df = pd.read_csv(args.pred, sep="\t")

    print(">>> Reading FASTA...")
    seq_dict = read_fasta(args.fasta)

    print(f">>> Total sequences loaded: {len(seq_dict)}")

    print(">>> Extracting SP features...")
    features = df.apply(lambda row: split_sp_regions(row, seq_dict), axis=1)

    print(">>> Merging results...")
    df = pd.concat([df, features], axis=1)

    print(">>> Saving output...")
    df.to_csv(args.out, sep="\t", index=False)

    print(">>> DONE.")


if __name__ == "__main__":
    main()