#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理73.86GB蛋白序列文件，过滤规则：
1. 序列长度 ≥ 30 aa
2. N端前70 aa 中 X 含量 ≤ 20%
3. 序列以 M 开头（排除截断序列）
"""
import gzip
import sys
import os
from collections import defaultdict
import time

# ====================== 配置参数 ======================
INPUT_FILE = "/home/wangjy/code/mgnify/mgy_clusters_FL1_only.fa.gz"
OUTPUT_FILE = "/home/wangjy/code/mgnify/cleaned_proteins_filtered.fa.gz"
LOG_FILE = "/home/wangjy/code/mgnify/filter_protein_python.log"
MIN_LENGTH = 30  # 最小序列长度
N_TERMINAL_LENGTH = 70  # N端检测长度
MAX_X_RATIO = 20.0  # X含量上限（%）

# ====================== 日志函数 ======================
def log(msg):
    """打印并记录日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

# ====================== FASTA 解析器（流式处理）=====================
def parse_fasta_gz(file_path):
    """
    流式解析gzip压缩的FASTA文件，逐行读取，不加载整个文件到内存
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    header = ""
    sequence = []
    
    # 流式读取gz文件
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 处理Windows换行符^M（\r）
            line = line.replace("\r", "")
            
            # 检测FASTA头部
            if line.startswith(">"):
                # 处理上一个序列
                if header and sequence:
                    yield header, "".join(sequence)
                # 初始化新序列
                header = line
                sequence = []
            else:
                # 拼接序列行
                sequence.append(line)
        
        # 处理最后一个序列
        if header and sequence:
            yield header, "".join(sequence)

# ====================== 过滤逻辑 ======================
def filter_sequence(seq):
    """
    过滤单个序列，返回是否保留
    """
    # 规则1：长度≥30
    if len(seq) < MIN_LENGTH:
        return False
    
    # 规则2：以M开头
    if not seq.startswith("M"):
        return False
    
    # 规则3：N端70aa的X含量≤20%
    n_terminal = seq[:N_TERMINAL_LENGTH]
    x_count = n_terminal.count("X")
    x_ratio = (x_count / len(n_terminal)) * 100 if len(n_terminal) > 0 else 100.0
    if x_ratio > MAX_X_RATIO:
        return False
    
    return True

# ====================== 主处理函数 ======================
def main():
    # 初始化日志
    open(LOG_FILE, "w", encoding="utf-8").close()  # 清空旧日志
    log(f"开始处理超大FASTA文件: {INPUT_FILE}")
    log(f"过滤规则：长度≥{MIN_LENGTH} | M开头 | N端{N_TERMINAL_LENGTH}aa X≤{MAX_X_RATIO}%")
    
    # 统计变量
    total_count = 0
    pass_count = 0
    fail_length = 0
    fail_no_M = 0
    fail_x_ratio = 0
    
    # 流式处理 + 写入结果
    try:
        with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as out_f:
            for header, seq in parse_fasta_gz(INPUT_FILE):
                total_count += 1
                
                # 过滤序列
                if filter_sequence(seq):
                    pass_count += 1
                    # 写入通过的序列
                    out_f.write(f"{header}\n{seq}\n")
                else:
                    # 统计失败原因
                    if len(seq) < MIN_LENGTH:
                        fail_length += 1
                    elif not seq.startswith("M"):
                        fail_no_M += 1
                    else:
                        n_terminal = seq[:N_TERMINAL_LENGTH]
                        x_ratio = (n_terminal.count("X") / len(n_terminal)) * 100
                        if x_ratio > MAX_X_RATIO:
                            fail_x_ratio += 1
                
                # 每处理10万条打印进度
                if total_count % 100000 == 0:
                    log(f"进度：已处理 {total_count:,} 条序列，通过 {pass_count:,} 条")
        
        # 输出统计结果
        log("="*50)
        log(f"处理完成！总计序列数：{total_count:,}")
        log(f"通过过滤：{pass_count:,} 条 ({pass_count/total_count*100:.2f}%)")
        log(f"失败原因：")
        log(f"  - 长度不足{MIN_LENGTH}aa：{fail_length:,} 条")
        log(f"  - 非M开头：{fail_no_M:,} 条")
        log(f"  - N端X含量超标：{fail_x_ratio:,} 条")
        log(f"输出文件：{OUTPUT_FILE}")
        
        # 验证输出文件大小
        if os.path.exists(OUTPUT_FILE):
            file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024*1024)
            log(f"输出文件大小：{file_size:.2f} GB")
        else:
            log("ERROR: 输出文件未生成！")
    
    except Exception as e:
        log(f"处理出错：{str(e)}")
        raise

if __name__ == "__main__":
    main()
