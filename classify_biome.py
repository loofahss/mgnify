import gzip
import sys

# === 配置输入输出 ===
INPUT_FILE = 'mgy_biomes.tsv.gz'

# 定义单项输出文件
OUTPUT_FILES = {
    'Human_Gut': 'id_human_gut.txt',
    'Soil':      'id_soil.txt',
    'Marine':    'id_marine.txt',
    'Freshwater':'id_freshwater.txt'
}

# 定义交集输出文件
INTERSECTION_FILE = 'id_intersections.tsv'

# === 定义生态位判定逻辑 (严格前缀) ===
PREFIXES = {
    'Human_Gut': 'root:host-associated:human:digestive system',
    'Soil':      'root:environmental:terrestrial:soil',
    'Marine':    'root:environmental:aquatic:marine',
    'Freshwater':'root:environmental:aquatic:freshwater'
}

def get_biome(lineage_str):
    """根据 lineage 判断属于哪个生态位，若都不匹配返回 None"""
    lineage = lineage_string.lower()
    for name, prefix in PREFIXES.items():
        if lineage.startswith(prefix):
            return name
    return None

def process_single_protein(prot_id, found_biomes, file_handles, intersect_handle):
    """
    处理单个蛋白的最终写入逻辑
    prot_id: 蛋白ID
    found_biomes: 该蛋白出现过的生态位集合 (set)
    file_handles: 单项文件句柄字典
    intersect_handle: 交集文件句柄
    """
    if not found_biomes:
        return

    # 1. 写入各自的单项文件 (允许重叠，只要出现就写入)
    for biome in found_biomes:
        file_handles[biome].write(f"{prot_id}\n")

    # 2. 如果出现超过1个生态位，写入交集文件
    if len(found_biomes) > 1:
        # 格式: ID <tab> Biome1,Biome2
        biomes_str = ",".join(sorted(list(found_biomes)))
        intersect_handle.write(f"{prot_id}\t{biomes_str}\n")

def main():
    print(f"正在处理 {INPUT_FILE} ...")
    
    # 打开所有文件句柄
    file_handles = {k: open(v, 'w') for k, v in OUTPUT_FILES.items()}
    intersect_handle = open(INTERSECTION_FILE, 'w')
    
    # 状态变量，用于聚合同一个ID的所有记录
    current_id = None
    current_biomes = set()
    
    line_count = 0
    
    try:
        with gzip.open(INPUT_FILE, 'rt') as f:
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"已扫描 {line_count} 行...", end='\r')

                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                prot_id = parts[0]
                lineage_raw = parts[2]
                
                # 判断当前行属于哪个生态位
                lineage = lineage_raw.lower()
                matched_biome = None
                for name, prefix in PREFIXES.items():
                    if lineage.startswith(prefix):
                        matched_biome = name
                        break
                
                # === 核心聚合逻辑 ===
                if prot_id != current_id:
                    # 发现新ID，先结算上一个ID的数据
                    if current_id is not None:
                        process_single_protein(current_id, current_biomes, file_handles, intersect_handle)
                    
                    # 重置状态，开始记录新ID
                    current_id = prot_id
                    current_biomes = set()
                    if matched_biome:
                        current_biomes.add(matched_biome)
                else:
                    # ID未变，继续累积生态位信息
                    if matched_biome:
                        current_biomes.add(matched_biome)

            # 循环结束，别忘了处理文件最后一行遗留的那个ID
            if current_id is not None:
                process_single_protein(current_id, current_biomes, file_handles, intersect_handle)
    
    finally:
        # 关闭所有文件
        for h in file_handles.values():
            h.close()
        intersect_handle.close()

    print(f"\n处理完成！共扫描 {line_count} 行。")
    print(f"结果已保存至: {list(OUTPUT_FILES.values())} 及 {INTERSECTION_FILE}")

if __name__ == '__main__':
    main()